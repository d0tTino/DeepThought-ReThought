"""Demonstrate planning from a natural language goal."""

from deepthought.planning import L2PTranslator, plan


def main() -> None:
    translator = L2PTranslator()
    domain, problem = translator.translate("move obj from loc1 to loc2")
    actions = plan(domain, problem)
    print("Plan:", actions)


if __name__ == "__main__":
    main()
