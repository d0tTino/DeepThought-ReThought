# Third-Party Licenses

This project relies on several open-source models and libraries. The table below
summarizes each component and its permissive license. All entries use
OSI-approved terms that allow commercial and derivative use with attribution.
The list reflects the components bundled with DeepThought-ReThought; consult the
upstream projects for the most current license information.

| Component | License | Notes |
| --- | --- | --- |
| PyTorch | BSD-3-Clause | Deep learning framework |
| Transformers | Apache-2.0 | Hugging Face model/optimizer library |
| Datasets | Apache-2.0 | Hugging Face dataset utilities |
| PEFT | Apache-2.0 | Parameter-Efficient Fine-Tuning |
| TRL | Apache-2.0 | Reinforcement learning helpers |
| Accelerate | Apache-2.0 | Hardware abstraction for transformers |
| bitsandbytes | MIT | 8-bit optimizers |
| SentencePiece | Apache-2.0 | Tokenizer library |
| Protobuf | BSD-3-Clause | Protocol buffers |
| SciPy | BSD-3-Clause | Scientific computing |
| Evaluate | Apache-2.0 | Model evaluation tools |
| nats-py | Apache-2.0 | NATS client |
| aiohttp | Apache-2.0 | Async HTTP client/server |
| FastAPI | MIT | Web framework |
| pydantic | MIT | Data validation |
| sentence-transformers | Apache-2.0 | Sentence embedding models |
| Uvicorn | BSD-3-Clause | ASGI server |
| PyYAML | MIT | YAML parser |
| FAISS | MIT | Vector similarity search |
| prometheus-client | Apache-2.0 | Metrics exporter |
| discord.py | MIT | Discord API client |
| NetworkX | BSD-3-Clause | Graph algorithms |
| TextBlob | MIT | NLP utilities |
| aiosqlite | MIT | Async SQLite interface |
| pytest-asyncio | Apache-2.0 | Async pytest support |
| VaderSentiment | MIT | Sentiment analysis |
| pyperplan | MIT | Automated planning |
| l2p | MIT | Learning to plan library |
| UME | MIT | Micro-embedding utilities |
| OpenCLIP (open-clip-torch) | MIT | CLIP implementation |
| torchvision | BSD-3-Clause | Computer vision models |
| opencv-python-headless | Apache-2.0 | Computer vision toolkit |
| BGE | MIT | Optional `PerceptionConfig.text_model` |
| E5 | MIT | Default `PerceptionConfig.text_model` |
| Default social perception model | MIT | Bundled classifier weights |
| WavLM | MIT | Speech representation model |
| WavLM checkpoints | MIT | Pretrained weights |
| CLAP | CC0 | Contrastive language-audio model |
| CLAP checkpoints | CC0 | Pretrained weights |
| SigLIP | Apache-2.0 | Vision-language model |
| SigLIP checkpoints | Apache-2.0 | Pretrained weights |

## License verification workflow

Encoder model defaults live in `src/deepthought/services/perception/config.py`.
When these defaults change, update `scripts/model_version_whitelist.json` with
the new versions and their corresponding license information. After updating the
whitelist and this document, verify everything with:

```
python scripts/check_model_versions.py
python scripts/verify_licenses.py
```

Both scripts will report an error if the versions or licenses diverge from the
whitelist or if required entries are missing from this table.
