from memory.facts_graph import FactsGraph


def test_sessions_increment(tmp_path):
    db = tmp_path / 'facts.db'
    fg = FactsGraph(str(db))
    key = 'default_user'
    info0 = fg.get_session_info(key)
    assert info0['session_count'] == 0
    fg.start_session(key)
    info1 = fg.get_session_info(key)
    assert info1['session_count'] == 1
    # A second start increments the counter
    fg.start_session(key)
    info2 = fg.get_session_info(key)
    assert info2['session_count'] == 2
    # Update session should not change session_count
    fg.update_session(key)
    info3 = fg.get_session_info(key)
    assert info3['session_count'] == 2
    fg.close()

