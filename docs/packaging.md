# Packaging and Publishing

This project consists of two Python distributions: the main `deepthought-rethought` package and a standalone `dtrt-finetune` training helper. Both are published to PyPI using [build](https://pypi.org/project/build/) and [twine](https://pypi.org/project/twine/).

## Build the Core Package

From the repository root run:

```bash
python -m build
```

The wheel will be created in `dist/`. Install it locally with:

```bash
pip install dist/*.whl
```

## Publish the Core Package

To upload the wheel to PyPI execute:

```bash
twine upload dist/*
```

Authentication is handled via the `PYPI_API_TOKEN` environment variable.

## Build and Publish `dtrt-finetune`

The training helper is defined by `train/pyproject.toml`. Use the helper script to build and upload it:

```bash
python tools/publish_finetune.py
```

The script builds the wheel under `train/dist/` and then runs `twine upload` on that file. Set `PYPI_API_TOKEN` in your environment to authenticate.
