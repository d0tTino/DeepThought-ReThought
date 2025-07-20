import sqlite3

import pytest

from deepthought.search.offline_search import OfflineSearch


def test_create_index_and_search(tmp_path):
    path = tmp_path / "index.db"
    docs = [("t1", "hello world"), ("t2", "goodbye world")]
    search = OfflineSearch.create_index(str(path), docs)
    assert path.exists()
    results = search.search("hello")
    assert results == ["hello world"]


def test_search_limit(tmp_path):
    path = tmp_path / "idx.db"
    docs = [("a", "alpha"), ("b", "beta"), ("c", "gamma")]
    search = OfflineSearch.create_index(str(path), docs)
    results = search.search("alpha OR beta", limit=2)
    assert results == ["alpha", "beta"]


def test_create_index_existing_db(tmp_path):
    """Calling ``create_index`` on an existing DB should not error."""
    path = tmp_path / "existing.db"
    docs1 = [("x", "foo")]
    OfflineSearch.create_index(str(path), docs1)

    docs2 = [("y", "bar")]
    # Should not raise when the database already contains the table
    search = OfflineSearch.create_index(str(path), docs2)

    results = search.search("foo OR bar", limit=5)
    assert "foo" in results
    assert "bar" in results


def test_search_expected_hits_and_limit(tmp_path):
    """Create an index and ensure search respects the limit."""
    path = tmp_path / "limit_hits.db"
    docs = [
        ("d1", "lorem ipsum"),
        ("d2", "ipsum lorem"),
        ("d3", "ipsum dolor"),
    ]
    search = OfflineSearch.create_index(str(path), docs)

    # limit should restrict the number of returned documents
    assert search.search("ipsum", limit=2) == ["lorem ipsum", "ipsum lorem"]

    # requesting more results should return all matching documents
    assert search.search("ipsum", limit=5) == [
        "lorem ipsum",
        "ipsum lorem",
        "ipsum dolor",
    ]


def test_create_index_empty_docs(tmp_path):
    path = tmp_path / "empty.db"
    with pytest.raises(ValueError):
        OfflineSearch.create_index(str(path), [])


def test_create_index_missing_fts5(monkeypatch, tmp_path):
    path = tmp_path / "nofts5.db"

    orig_connect = sqlite3.connect

    class FailingConn(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE VIRTUAL TABLE" in sql:
                raise sqlite3.OperationalError("no such module: fts5")
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        kwargs["factory"] = FailingConn
        return orig_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)

    with pytest.raises(RuntimeError, match="FTS5"):
        OfflineSearch.create_index(str(path), [("t", "c")])
