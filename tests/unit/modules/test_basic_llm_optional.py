import pytest
from deepthought import modules


def test_basic_llm_instantiation():
    """Instantiate BasicLLM or skip if optional deps are missing."""
    BasicLLM = modules.BasicLLM
    try:
        instance = BasicLLM(None, None)
    except ImportError:
        pytest.skip("BasicLLM dependencies not installed")
    assert instance is not None
