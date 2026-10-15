"""
Tests for mine_patterns_tool in tools_patterns.py matching TEST_FIXTURE.md known answers #17 and #20.
"""

import json
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _setup_fixture_session():
    """Helper to initialize a fresh session with sample_fixture.csv loaded."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_known_answer_17_tfidf_fpgrowth():
    """Test known answer #17: tfidf filtering + fpgrowth algorithm."""
    session, mine_patterns_tool = _setup_fixture_session()

    # catA, min_sup=3 -> {alpha} support 3
    res_a3 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    json.dumps(res_a3, allow_nan=False)
    assert res_a3.get("num_patterns") == 1
    assert res_a3["patterns"][0]["pattern"] == ["alpha"]
    assert res_a3["patterns"][0]["support"] == 3

    # catA, min_sup=4 -> 0 patterns
    res_a4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    json.dumps(res_a4, allow_nan=False)
    assert res_a4.get("num_patterns") == 0
    assert res_a4["patterns"] == []

    # catB, min_sup=4 -> {gamma} support 4
    res_b4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    json.dumps(res_b4, allow_nan=False)
    assert res_b4.get("num_patterns") == 1
    assert res_b4["patterns"][0]["pattern"] == ["gamma"]
    assert res_b4["patterns"][0]["support"] == 4


def test_mine_patterns_zero_surviving_terms():
    """Test zero surviving terms for variance and term_frequency methods."""
    session, mine_patterns_tool = _setup_fixture_session()

    for category in ["catA", "catB"]:
        for f_method in ["variance", "term_frequency"]:
            # fpgrowth
            res = mine_patterns_tool.invoke({
                "category_name": category,
                "filtering_method": f_method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            json.dumps(res, allow_nan=False)
            assert res.get("num_patterns") == 0
            assert res["patterns"] == []
            assert "status" not in res or res.get("status") != "error"

            # topk
            res_topk = mine_patterns_tool.invoke({
                "category_name": category,
                "filtering_method": f_method,
                "algorithm": "topk",
                "k": 1
            })
            json.dumps(res_topk, allow_nan=False)
            assert res_topk.get("num_patterns") == 0
            assert res_topk["patterns"] == []
            assert "status" not in res_topk or res_topk.get("status") != "error"


def test_mine_patterns_known_answer_20_topk_and_maxfpgrowth():
    """Test known answer #20: topk (FAE) and maxfpgrowth algorithms."""
    session, mine_patterns_tool = _setup_fixture_session()

    # algorithm="topk" for k=1, 2, 3
    for k_val in [1, 2, 3]:
        # catA -> {alpha} support 3
        res_a = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        json.dumps(res_a, allow_nan=False)
        assert res_a.get("num_patterns") == 1
        assert res_a["patterns"][0]["pattern"] == ["alpha"]
        assert res_a["patterns"][0]["support"] == 3

        # catB -> {gamma} support 4
        res_b = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        json.dumps(res_b, allow_nan=False)
        assert res_b.get("num_patterns") == 1
        assert res_b["patterns"][0]["pattern"] == ["gamma"]
        assert res_b["patterns"][0]["support"] == 4

    # algorithm="maxfpgrowth" for min_sup=1 and 4 -> returns 0 patterns for both categories
    for min_sup_val in [1, 4]:
        for category in ["catA", "catB"]:
            res_max = mine_patterns_tool.invoke({
                "category_name": category,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup_val
            })
            json.dumps(res_max, allow_nan=False)
            assert res_max.get("num_patterns") == 0
            assert res_max["patterns"] == []


def test_mine_patterns_error_cases():
    """Test error handling for invalid parameters and non-existent category."""
    session, mine_patterns_tool = _setup_fixture_session()

    # Non-existent category
    res_cat = mine_patterns_tool.invoke({
        "category_name": "catZ",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    json.dumps(res_cat, allow_nan=False)
    assert res_cat.get("status") == "error"
    assert "catZ" in res_cat.get("message", "")

    # fpgrowth without min_sup
    res_no_sup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    json.dumps(res_no_sup, allow_nan=False)
    assert res_no_sup.get("status") == "error"
    assert "min_sup" in res_no_sup.get("message", "")

    # topk without k
    res_no_k = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    json.dumps(res_no_k, allow_nan=False)
    assert res_no_k.get("status") == "error"
    assert "k" in res_no_k.get("message", "")

    # topk with k=0
    res_k0 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 0
    })
    json.dumps(res_k0, allow_nan=False)
    assert res_k0.get("status") == "error"
    assert "k" in res_k0.get("message", "")
