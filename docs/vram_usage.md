# VRAM Usage for 3B Models

This guide summarizes typical memory requirements when working with 3B parameter models.
The numbers are based on the output of `dtrt finetune --estimate-vram` and running the
quantized models for inference.

```bash
dtrt finetune --estimate-vram --model-path meta-llama/Llama-3.2-3B-Instruct
```

Example output:

```
Estimated VRAM requirement: 6.4 GB
```

## Summary

The table below lists approximate VRAM needs for a 3B model under common scenarios.
These values serve as a general guideline and may vary depending on batch size
and optimizer settings.

| Scenario | Approx. VRAM | Notes |
|---------|-------------|-------|
| QLoRA fine-tuning | ~6–7 GB | Matches the estimate shown above |
| 4-bit inference | ~3–4 GB | Using quantized weights (e.g., AWQ or bitsandbytes) |

Fine-tuning requires additional RAM for activations and optimizer state, hence the
higher usage compared to inference. The `--estimate-vram` flag helps gauge whether
your GPU has sufficient capacity before launching training.
