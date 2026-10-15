import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools

def test_variance_filter_tool_known_answer_5():
    # 1. Initialize session
    session = SessionState()

    # 2. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Import build_dtm_tool and run it to construct session.feature_matrix
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Initialize filtering tools
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 5. Run variance_filter_tool with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # Read back full_report stored in session
    stored_res = session.results_store.get(res["result_id"])
    full_report = stored_res["full_report"]

    # Spec checks:
    # Check 1: Column 1 is term/feature name, Column 2 is variance
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # Check 2: Values are sorted in descending order by variance (1st row must be 'alpha')
    assert full_report.iloc[0]["term"] == "alpha"
    assert full_report["variance"].is_monotonic_decreasing

    # Map terms to their computed variance
    term_variances = dict(zip(full_report["term"], full_report["variance"]))

    # Known answer 5 expected variances
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375
    }

    # Verify each term's computed variance against expected values
    for term, expected_var in expected_variances.items():
        assert term in term_variances, f"Term '{term}' missing from full_report"
        assert term_variances[term] == pytest.approx(expected_var, abs=1e-3)

    # Verify thresholding behavior (threshold=0.15 should remove 'always' and keep the rest)
    assert res["n_kept_terms"] == 4
    assert res["n_removed_terms"] == 1

    kept_terms = set(session.artifacts["variance_kept_terms"])
    assert kept_terms == {"alpha", "beta", "delta", "gamma"}
    assert "always" not in kept_terms


def test_variance_filter_strict_greater_threshold():
    # Verify strict greater than behavior (variance > threshold, so variance == threshold is removed)
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Set threshold exactly equal to 'always''s variance: 0.109375
    res = variance_filter_tool.invoke({"threshold": 0.109375})

    # 'always' has variance equal to threshold (0.109375), so strictly > threshold MUST remove it
    assert res["n_kept_terms"] == 4
    assert res["n_removed_terms"] == 1

    kept_terms = set(session.artifacts["variance_kept_terms"])
    assert "always" not in kept_terms
    assert kept_terms == {"alpha", "beta", "delta", "gamma"}


def setup_session_with_dtm():
    """Helper to initialize session with sample fixture dataset and build DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_variance_filter_tool_empty_matrix():
    """Edge case: Run variance filter when feature matrix is missing."""
    session = SessionState()
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    result = variance_filter_tool.invoke({"threshold": 0.15})
    assert result["status"] == "error"


def test_pearson_filter_tool_known_answer_6():
    """Known Answer 6: Check pearson_filter_tool on target_class='catB'."""
    session = setup_session_with_dtm()
    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    summary = pearson_filter_tool.invoke({"target_class": "catB"})
    
    assert summary["target_class"] == "catB"
    assert summary["n_total_features"] == 5
    
    # Retrieve full_report and verify column names
    full_report = session.results_store[summary["result_id"]]["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]
    
    # Verify Known Answer 6 values
    r_map = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert r_map["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_map["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_map["gamma"] == pytest.approx(0.8307, abs=1e-3)
    
    # Verify exact ordering by absolute correlation descending (|1.0| > |0.8307| > |-0.7746| > |-0.6882| > |0.3780|)
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert full_report["term"].tolist() == expected_order


def test_pearson_filter_tool_invalid_target_class():
    """Edge case: Request a non-existent target_class."""
    session = setup_session_with_dtm()
    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    result = pearson_filter_tool.invoke({"target_class": "non_existent_category"})
    assert result["status"] == "error"
    assert "available_categories" in result


def test_pearson_filter_tool_constant_column():
    """Edge case: Drop empty text row before building DTM so 'always' becomes constant."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Remove empty text row BEFORE building DTM
    session.dataframe = session.dataframe[session.dataframe["text"].str.strip() != ""].copy()
    session.labels = session.dataframe["category_name"].values
    
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    summary = pearson_filter_tool.invoke({"target_class": "catB"})
    full_report = session.results_store[summary["result_id"]]["full_report"]
    r_map = dict(zip(full_report["term"], full_report["pearson_r"]))
    
    # 'always' is now present in all remaining 7 rows (constant column)
    assert not np.isnan(r_map["always"])
    assert r_map["always"] == pytest.approx(0.0, abs=1e-6)


def test_variance_filter_tool_floating_point_precision_issue():
    """Bug reproduction: Drop empty row so 'always' variance is theoretically 0.0.
    E[x^2] - (E[x])^2 gives ~2.2e-16 due to floating point imprecision, causing
    threshold=0.0 to erroneously keep 'always'.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Remove empty text row BEFORE building DTM so 'always' count is 1 in all 7 docs
    session.dataframe = session.dataframe[session.dataframe["text"].str.strip() != ""].copy()
    session.labels = session.dataframe["category_name"].values
    
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    res = variance_filter_tool.invoke({"threshold": 0.0})
    kept_terms = set(session.artifacts["variance_kept_terms"])
    
    # 'always' should have 0 variance and be removed when threshold=0.0
    assert "always" not in kept_terms
    assert res["n_kept_terms"] == 4


def test_spearman_filter_tool_known_answer_6():
    """Known Answer 6: Check spearman_filter_tool on target_class='catB'."""
    session = setup_session_with_dtm()
    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    summary = spearman_filter_tool.invoke({"target_class": "catB"})

    assert summary["target_class"] == "catB"
    assert summary["n_total_features"] == 5

    # Retrieve full_report and verify column names
    full_report = session.results_store[summary["result_id"]]["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Verify Known Answer 6 Spearman values
    r_map = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert r_map["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_map["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_map["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Verify exact ordering by absolute correlation descending (|1.0000| > |0.9363| > |-0.7746| > |-0.7500| > |0.3780|)
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert full_report["term"].tolist() == expected_order


def test_spearman_filter_tool_invalid_target_class():
    """Edge case: Request a non-existent target_class for spearman_filter_tool."""
    session = setup_session_with_dtm()
    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result = spearman_filter_tool.invoke({"target_class": "non_existent_category"})
    assert result["status"] == "error"
    assert "available_categories" in result


def test_spearman_filter_tool_constant_column():
    """Edge case: Drop empty text row so 'always' becomes constant for spearman_filter_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Remove empty text row BEFORE building DTM
    session.dataframe = session.dataframe[session.dataframe["text"].str.strip() != ""].copy()
    session.labels = session.dataframe["category_name"].values

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    summary = spearman_filter_tool.invoke({"target_class": "catB"})
    full_report = session.results_store[summary["result_id"]]["full_report"]
    r_map = dict(zip(full_report["term"], full_report["spearman_r"]))

    # 'always' is now present in all remaining 7 rows (constant column)
    assert not np.isnan(r_map["always"])
    assert r_map["always"] == pytest.approx(0.0, abs=1e-6)
