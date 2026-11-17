"""Checks that /search still answers in the agreed format.

Run before each push, from the 02_model_server/ folder:
    python tests/test_search.py
(or `pytest tests/` if pytest is installed). No server needs to be running:
the test calls the search function in app.py directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import app` finds 02_model_server/app.py

import gradio as gr
from app import search

SCORE_KEYS = {"circulation_score", "daylight_score", "cross_vent_rooms", "dark_rooms", "missing_door"}
PLAN_KEYS = {"id", "bedrooms", "bathrooms", "area_sqft", "area_diff_sqft", "room_counts", "relation", "scores"}
ROOM_TYPES = ["living", "bedroom", "kitchen", "bathroom", "balcony", "storage", "stair", "front_door"]


def check_format(result):
    assert set(result) == {"query", "room_types", "count", "plans"}, f"top-level keys changed: {set(result)}"
    assert result["room_types"] == ROOM_TYPES, "room type order changed (the matrix rows depend on it)"
    assert result["count"] == len(result["plans"]) == 6, f"expected 6 plans, got {len(result['plans'])}"
    ids = [p["id"] for p in result["plans"]]
    assert len(set(ids)) == 6, f"a plan appears twice: {ids}"
    for p in result["plans"]:
        assert set(p) == PLAN_KEYS, f"plan keys changed: {set(p)}"
        assert set(p["scores"]) == SCORE_KEYS, f"score keys changed: {set(p['scores'])}"
        assert list(p["room_counts"]) == ROOM_TYPES
        assert len(p["relation"]) == 8 and all(len(row) == 8 for row in p["relation"]), "relation is not 8 x 8"
        assert 0 <= p["scores"]["daylight_score"] <= 1 and 0 <= p["scores"]["circulation_score"] <= 1


def test_common_query():
    result = search(3, 1100)
    check_format(result)
    for p in result["plans"]:
        assert p["bedrooms"] == 3, "a common query must return the exact bedroom count"
        assert not p["scores"]["missing_door"], "a common query must prefer plans without a missing door"
        assert abs(p["area_diff_sqft"]) <= 110, "a common query must stay within 10% of the area"
    # fixed order, the same on every computer: closest area, then higher daylight, then lower id
    keys = [(abs(p["area_diff_sqft"]), -p["scores"]["daylight_score"], p["id"]) for p in result["plans"]]
    assert keys == sorted(keys), f"plans are not in the fixed order: {keys}"
    assert search(3, 1100) == result, "the same query must give the same answer"


def test_rare_query_still_gives_6():
    check_format(search(7, 1000))     # only 1 plan has 7 bedrooms: the rest come from relaxed filters


def test_bad_input_is_rejected():
    for bedrooms, sqft in [(3, 50), (3, 50000), (-1, 1000), (20, 1000)]:
        try:
            search(bedrooms, sqft)
        except gr.Error:
            continue
        raise AssertionError(f"search({bedrooms}, {sqft}) should raise an error")


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"all {len(tests)} tests passed")
