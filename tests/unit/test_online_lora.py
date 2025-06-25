from deepthought.learn.online_lora import OnlineLoRALearner


def test_record_and_retrieve():
    learner = OnlineLoRALearner()
    learner.record_interaction("guild", "hi", "hello", 1.0)
    data = learner.get_training_data("guild")
    assert len(data) == 1
    assert data[0].prompt == "hi"
    assert data[0].response == "hello"
    assert data[0].reward == 1.0
