import pytest
from deepthought import modules


def test_basic_llm_placeholder():
    BasicLLM = modules.BasicLLM
    if BasicLLM.__module__ != "deepthought.modules.llm_basic":
        with pytest.raises(ImportError):
            BasicLLM(None, None)
    else:
        pytest.skip("BasicLLM dependencies available; placeholder not used")
