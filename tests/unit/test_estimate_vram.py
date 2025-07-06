import importlib
import sys
import types

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = object

    def load_dataset(*args, **kwargs):
        raise RuntimeError("load_dataset called")

    datasets_stub.load_dataset = load_dataset
    sys.modules["datasets"] = datasets_stub

if "peft" not in sys.modules:
    peft_stub = types.ModuleType("peft")
    peft_stub.LoraConfig = object
    peft_stub.get_peft_model = lambda *a, **k: None
    peft_stub.prepare_model_for_kbit_training = lambda m: m
    sys.modules["peft"] = peft_stub


def test_estimate_vram_simple():
    train = importlib.import_module("deepthought.train")
    from transformers import AutoModelForCausalLM, GPT2Config

    cfg = GPT2Config(n_embd=4, n_layer=1, n_head=1, vocab_size=10)
    model = AutoModelForCausalLM.from_config(cfg)
    vram = train.estimate_vram(model, batch_size=1, seq_length=2, bits=4)
    assert vram > 0
