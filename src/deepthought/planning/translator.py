from __future__ import annotations

"""Very small NL -> PDDL translator using ``l2p`` utilities."""

import tempfile
from typing import Tuple

from l2p.utils import parse_domain, parse_problem

_MOVE_DOMAIN = """
(define (domain move)
  (:predicates (at ?o ?l) (connected ?l1 ?l2))
  (:action move
     :parameters (?o ?from ?to)
     :precondition (and (at ?o ?from) (connected ?from ?to))
     :effect (and (not (at ?o ?from)) (at ?o ?to))
  )
)
"""

_MOVE_PROBLEM = """
(define (problem move-task)
  (:domain move)
  (:objects {obj} {frm} {to})
  (:init (at {obj} {frm}) (connected {frm} {to}))
  (:goal (and (at {obj} {to})))
)
"""


class L2PTranslator:
    """Translate very simple movement goals to PDDL."""

    def translate(self, goal: str) -> Tuple[str, str]:
        """Return ``(domain_pddl, problem_pddl)`` for ``goal``."""
        text = goal.lower()
        if not text.startswith("move"):
            raise ValueError("Unsupported goal: %s" % goal)
        parts = text.split()
        try:
            obj = parts[1]
            frm = parts[parts.index("from") + 1]
            to = parts[parts.index("to") + 1]
        except Exception as exc:  # pragma: no cover - invalid input
            raise ValueError("Goal must be of the form 'move X from A to B'") from exc
        domain = _MOVE_DOMAIN.strip()
        problem = _MOVE_PROBLEM.format(obj=obj, frm=frm, to=to).strip()
        # Validate using l2p's PDDL parser
        with tempfile.NamedTemporaryFile("w", delete=False) as df:
            df.write(domain)
            dpath = df.name
        with tempfile.NamedTemporaryFile("w", delete=False) as pf:
            pf.write(problem)
            ppath = pf.name
        parse_domain(dpath)
        parse_problem(ppath)
        return domain, problem
