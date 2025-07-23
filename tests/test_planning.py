from deepthought.planning import L2PTranslator, plan


def test_planning_move():
    translator = L2PTranslator()
    domain, problem = translator.translate("move obj from loc1 to loc2")
    actions = plan(domain, problem)
    assert actions == ["(move obj loc1 loc2)"]
