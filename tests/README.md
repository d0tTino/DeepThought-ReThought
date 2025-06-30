# Test Suite

The automated tests rely on the packages listed in `requirements-ci.txt`. You can install them with:

```bash
pip install -r requirements-ci.txt
```

A few integration tests require additional optional dependencies. These tests are skipped automatically when the packages are missing:

- `chromadb` for `test_chroma_service.py` (requires a running Chroma service)
- `pymemgraph` for `test_memgraph_service.py` (requires a Memgraph instance)
- `deepthought.motivate` for `test_motivation.py` and `test_reward_manager.py`

Install these extras if you want to run the entire suite:

```bash
pip install chromadb pymemgraph deepthought-motivate
```
