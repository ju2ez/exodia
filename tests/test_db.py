from exodia import db


def test_vote_cast_change_retract(tmp_path):
    p = tmp_path / "c.db"
    db.init_db(p)
    assert db.cast_vote(p, "a", "v1", 1) == {"idea_id": "a", "up": 1, "down": 0, "score": 1}
    db.cast_vote(p, "a", "v2", 1)
    t = db.cast_vote(p, "a", "v2", -1)  # v2 changes its mind
    assert (t["up"], t["down"], t["score"]) == (1, 1, 0)
    t = db.cast_vote(p, "a", "v1", 0)  # v1 retracts
    assert (t["up"], t["down"]) == (0, 1)
    assert db.voter_votes(p, "v2") == {"a": -1}


def test_tallies_zero_fill(tmp_path):
    p = tmp_path / "c.db"
    db.init_db(p)
    db.cast_vote(p, "a", "v", 1)
    t = db.tallies(p, ["a", "b"])
    assert t["a"]["score"] == 1
    assert t["b"] == {"up": 0, "down": 0, "score": 0}


def test_feedback_add_list(tmp_path):
    p = tmp_path / "c.db"
    db.init_db(p)
    f = db.add_feedback(p, "a", "great idea", "alice")
    assert f["id"] and f["author"] == "alice"
    db.add_feedback(p, "a", "more", "bob")
    lst = db.list_feedback(p, "a")
    assert len(lst) == 2 and lst[0]["text"] == "great idea"
    assert db.feedback_by_idea(p)["a"][1]["author"] == "bob"
