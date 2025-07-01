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
