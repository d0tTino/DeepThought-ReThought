from deepthought.cli import _build_parser


def test_parse_orchestrate_args():
    parser = _build_parser()
    args = parser.parse_args(["orchestrate", "cfg.yml"])
    assert args.command == "orchestrate"
    assert args.config == "cfg.yml"
    assert args.func.__name__ == "_cmd_orchestrate"
