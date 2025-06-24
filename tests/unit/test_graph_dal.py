import pytest
from deepthought.graph.dal import GraphDAL


class DummyConnector:
    def __init__(self, result=None):
        self.executed = []
        self.result = result if result is not None else []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        return self.result


def test_add_entity():
    connector = DummyConnector()
    dal = GraphDAL(connector)
    dal.add_entity("Person", {"name": "Alice"})

    assert connector.executed == [
        ("MERGE (n:Person $props)", {"props": {"name": "Alice"}})
    ]


def test_add_relationship():
    connector = DummyConnector()
    dal = GraphDAL(connector)
    dal.add_relationship(1, 2, "KNOWS", {"since": 2020})

    assert connector.executed == [
        (
            "MATCH (a {id: $start_id}), (b {id: $end_id}) MERGE (a)-[r:KNOWS $props]->(b)",
            {"start_id": 1, "end_id": 2, "props": {"since": 2020}},
        )
    ]


def test_add_entity_invalid_label():
    connector = DummyConnector()
    dal = GraphDAL(connector)

    with pytest.raises(ValueError):
        dal.add_entity("Person;MATCH(n)", {"name": "Alice"})

    assert connector.executed == []


def test_add_relationship_invalid_type():
    connector = DummyConnector()
    dal = GraphDAL(connector)

    with pytest.raises(ValueError):
        dal.add_relationship(1, 2, "KNOWS DELETE", {"since": 2020})

    assert connector.executed == []


def test_get_entity():
    connector = DummyConnector(result=[{"name": "Alice"}])
    dal = GraphDAL(connector)
    result = dal.get_entity("Person", "name", "Alice")

    assert result == {"name": "Alice"}
    assert connector.executed == [
        ("MATCH (n:Person {name: $value}) RETURN n", {"value": "Alice"})
    ]


def test_query_subgraph():
    connector = DummyConnector(result=[{"id": 1}])
    dal = GraphDAL(connector)
    q = "MATCH (n) RETURN n LIMIT 1"
    result = dal.query_subgraph(q, {"limit": 1})

    assert result == [{"id": 1}]
    assert connector.executed == [(q, {"limit": 1})]
