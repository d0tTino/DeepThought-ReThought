# Code Generation Service

The **CodeGenerationService** provides a minimal example of template based code
generation. It listens for `dtr.codegen.template_request` events and publishes
the executed result on `dtr.codegen.generated`.

A request payload should look like:

```json
{
  "template": "result = ${x} + ${y}",
  "variables": {"x": 1, "y": 2},
  "input_id": "42"
}
```

The service substitutes the variables into the template, executes the generated
code and publishes a `CodeGeneratedPayload` containing the final code and result.

This example is intentionally simple and intended as a starting point for more
advanced JIT or template based generation experiments.

