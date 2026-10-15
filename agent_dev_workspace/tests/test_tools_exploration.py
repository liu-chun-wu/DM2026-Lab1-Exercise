import json
import math
import matplotlib.figure
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_feature_correlation_matrix_known_answer_19():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # 1. Load dataset
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 2. Build DTM
    from tools_dtm import make_tools as make_dtm_tools

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Test feature_correlation_matrix_tool
    from tools_exploration import make_tools as make_exploration_tools

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    res = feature_corr_tool.invoke({})

    # Known answer 19 checks
    expected_top_features = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_features"] == expected_top_features
    assert res["n_features"] == 5

    expected_matrix = np.array(
        [
            [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
            [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
            [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
            [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
            [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
        ]
    )

    actual_matrix = np.array(res["correlation_matrix"])

    # Diagonal must be 1.0
    np.testing.assert_allclose(np.diag(actual_matrix), 1.0, atol=1e-3)

    # Full matrix check
    np.testing.assert_allclose(actual_matrix, expected_matrix, atol=1e-3)

    # Check pending_figure is a matplotlib Figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)


def test_feature_correlation_matrix_constant_column_json_serializable():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # 1. Load dataset
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 2. Drop the empty text row (row 7) so 'always' becomes a constant term in every remaining row
    session.dataframe = session.dataframe[
        session.dataframe["text"].str.strip() != ""
    ].reset_index(drop=True)

    # 3. Build DTM
    from tools_dtm import make_tools as make_dtm_tools

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Run feature_correlation_matrix_tool
    from tools_exploration import make_tools as make_exploration_tools

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    res = feature_corr_tool.invoke({})

    # 5. Must be strictly JSON-serializable without NaN (allow_nan=False raises ValueError if NaN exists)
    serialized = json.dumps(res, allow_nan=False)
    assert isinstance(serialized, str)


def test_feature_correlation_matrix_no_dtm_error():
    session = SessionState()
    from tools_exploration import make_tools as make_exploration_tools

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    res = feature_corr_tool.invoke({})
    assert res.get("status") == "error"
    assert "No feature matrix found" in res.get("message", "")


def test_feature_correlation_matrix_invalid_top_k():
    session = SessionState()
    from tools_exploration import make_tools as make_exploration_tools

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    for invalid_k in [-1, 0, 31]:
        res = feature_corr_tool.invoke({"top_k": invalid_k})
        assert res.get("status") == "error"
        assert "must be between 1 and 30" in res.get("message", "")


def test_cosine_similarity_known_answer_16():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # 1. Load dataset
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 2. Build DTM
    from tools_dtm import make_tools as make_dtm_tools

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Test cosine_similarity_tool
    from tools_exploration import make_tools as make_exploration_tools

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    # Check known answer 16 pairs
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Empty text row vs document 0 must be exactly 0.0 and not NaN
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["similarity"] == 0.0
    assert not math.isnan(res_0_7["similarity"])


def test_cosine_similarity_error_handling():
    session = SessionState()
    from tools_exploration import make_tools as make_exploration_tools

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    # 1. Missing DTM / dataframe
    res_no_dtm = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_no_dtm.get("status") == "error"
    assert "not found in session" in res_no_dtm.get("message", "")

    # Setup session with dataset and DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    from tools_dtm import make_tools as make_dtm_tools

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 2. Out-of-bounds indices (-1 and 8)
    for invalid_idx in [-1, 8]:
        res_invalid_1 = cosine_sim_tool.invoke(
            {"doc_index_1": invalid_idx, "doc_index_2": 0}
        )
        assert res_invalid_1.get("status") == "error"
        assert f"doc_index_1 must be between 0 and 7" in res_invalid_1.get(
            "message", ""
        )

        res_invalid_2 = cosine_sim_tool.invoke(
            {"doc_index_1": 0, "doc_index_2": invalid_idx}
        )
        assert res_invalid_2.get("status") == "error"
        assert f"doc_index_2 must be between 0 and 7" in res_invalid_2.get(
            "message", ""
        )
