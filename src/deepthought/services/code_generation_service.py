import ast
import asyncio
import json
import logging
import operator
from datetime import datetime, timezone
from string import Template
from typing import Any, Dict

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import (
    CodeGeneratedPayload,
    EventSubjects,
)
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class CodeGenerationService:
    """Simple template-based code generation service."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)

    _bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def _eval_expr(self, node: ast.AST, variables: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"Unknown variable '{node.id}' in expression")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._bin_ops:
                raise ValueError("Unsupported operator")
            left = self._eval_expr(node.left, variables)
            right = self._eval_expr(node.right, variables)
            return self._bin_ops[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._unary_ops:
                raise ValueError("Unsupported unary operator")
            operand = self._eval_expr(node.operand, variables)
            return self._unary_ops[op_type](operand)
        raise ValueError("Disallowed expression")

    async def _safe_execute(self, code: str, variables: Dict[str, Any], timeout: float = 0.1) -> Any:
        """Safely evaluate a simple assignment expression.

        The code must consist of a single assignment to ``result`` and may only
        contain arithmetic expressions built from constants or provided
        variables. Loops and excessively large constants are rejected. The
        evaluation runs in a separate thread and is cancelled if it exceeds the
        given ``timeout`` seconds.
        """

        def check_ast(module: ast.Module) -> None:
            for node in ast.walk(module):
                if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.AsyncWith, ast.With)):
                    raise ValueError("Loops are not allowed")
                if isinstance(node, ast.Constant):
                    val = node.value
                    if isinstance(val, (int, float)) and abs(val) > 1_000_000:
                        raise ValueError("Constant too large")
                    if isinstance(val, str) and len(val) > 1000:
                        raise ValueError("Constant too large")

        def eval_ast() -> Any:
            module = ast.parse(code, mode="exec")
            check_ast(module)
            if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
                raise ValueError("Code must be a single assignment to 'result'")
            assign = module.body[0]
            if (
                len(assign.targets) != 1
                or not isinstance(assign.targets[0], ast.Name)
                or assign.targets[0].id != "result"
            ):
                raise ValueError("Code must assign to 'result'")
            return self._eval_expr(assign.value, variables)

        return await asyncio.wait_for(asyncio.to_thread(eval_ast), timeout)

    async def _handle_template_request(self, msg: Msg) -> None:
        request_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("CodeTemplate payload must be a dict")
            request_id = data.get("input_id")
            template_text = data.get("template")
            variables = data.get("variables", {})
            if not isinstance(template_text, str) or not isinstance(variables, dict):
                raise ValueError("Invalid template payload fields")
            logger.info("CodeGenerationService received request ID %s", request_id)

            code = Template(template_text).safe_substitute(**variables)
            result = await self._safe_execute(code, variables)

            payload = CodeGeneratedPayload(
                code=code,
                result=str(result),
                input_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            await self._publisher.publish(
                EventSubjects.CODE_GENERATED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("CodeGenerationService published generated code event ID %s", request_id)
            await msg.ack()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid CodeTemplate payload: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)
        except Exception as e:  # pragma: no cover - unexpected failures
            logger.error("Error in CodeGenerationService handler: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def start(self, durable_name: str = "codegen_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for CodeGenerationService.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.CODE_TEMPLATE_REQUEST,
                handler=self._handle_template_request,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("CodeGenerationService subscribed to %s", EventSubjects.CODE_TEMPLATE_REQUEST)
            return True
        except nats.errors.Error as e:
            logger.error("CodeGenerationService failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("CodeGenerationService failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("CodeGenerationService stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")

    async def __aenter__(self) -> "CodeGenerationService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
