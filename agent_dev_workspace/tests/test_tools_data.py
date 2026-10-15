import json
import numpy as np
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool_known_answer_10():
    """
    Test tokenize_tool against TEST_FIXTURE.md known answer 10 using sample_fixture.csv.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tool
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    # Execute tool
    result = tokenize_tool.invoke({})
    assert result["status"] == "success"
    assert result["n_documents"] == 8

    # Expected token lists per TEST_FIXTURE.md Known Answer 10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    actual_tokens = session.dataframe["unigrams"].tolist()
    assert actual_tokens == expected_tokens


def test_tokenize_tool_uppercase_and_punctuation():
    """
    Test that tokenize_tool correctly lowercases text and handles punctuation with nltk.word_tokenize.
    """
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["Hello, World!"]})

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})
    assert result["status"] == "success"

    actual_tokens = session.dataframe["unigrams"].iloc[0]
    expected_tokens = ["hello", ",", "world", "!"]
    assert actual_tokens == expected_tokens


def test_tokenize_tool_no_dataframe_error():
    """
    Test that tokenize_tool returns an error status when no dataframe is loaded in session.
    """
    session = SessionState()
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})
    assert result["status"] == "error"
    assert "No dataset loaded" in result["message"]


def test_list_files_tool_known_answer_11():
    """
    Test list_files_tool against TEST_FIXTURE.md known answer 11 using subdirectory='newdataset'.
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert result["status"] == "success"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_security_and_errors():
    """
    Test security constraints (preventing repo escapes) and nonexistent path error handling.
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Path traversal attempt with "../"
    res_parent = list_files_tool.invoke({"subdirectory": "../"})
    assert res_parent["status"] == "error"

    # Absolute path escape with "/etc"
    res_etc = list_files_tool.invoke({"subdirectory": "/etc"})
    assert res_etc["status"] == "error"

    # Non-existent directory
    res_nonexistent = list_files_tool.invoke({"subdirectory": "non_existent_folder_xyz"})
    assert res_nonexistent["status"] == "error"


def test_list_files_tool_sorting():
    """
    Test that list_files_tool returns files and directories in sorted order.
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "agent_dev"})
    assert result["status"] == "success"
    assert result["files"] == sorted(result["files"])
    assert result["directories"] == sorted(result["directories"])


def test_check_missing_tool_known_answer_2():
    """
    Test check_missing_tool against TEST_FIXTURE.md known answer 2 using sample_fixture.csv.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})
    assert result["status"] == "success"
    assert result["n_documents"] == 8
    assert result["n_missing"] == 1
    assert result["missing_indices"] == [7]


def test_check_missing_tool_edge_cases():
    """
    Test check_missing_tool on custom dataframe containing NaN, None, empty string, and whitespace string.
    Ensures dataframe is not mutated.
    """
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["ok", "", " ", None, "a"]})

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})
    assert result["status"] == "success"
    assert result["n_documents"] == 5
    assert result["n_missing"] == 3
    assert result["missing_indices"] == [1, 2, 3]

    # Verify session.dataframe length and content remains untouched
    assert len(session.dataframe) == 5


def test_check_missing_tool_no_dataframe_error():
    """
    Test that check_missing_tool returns an error status when no dataframe is loaded.
    """
    session = SessionState()
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})
    assert result["status"] == "error"
    assert "No dataset loaded" in result["message"]


def test_check_duplicates_tool_known_answer_3():
    """
    Test check_duplicates_tool against TEST_FIXTURE.md known answer 3 using sample_fixture.csv.
    Verify default call invoke({}) reports 1 duplicate with 8 remaining rows, and drop=True removes 2 rows leaving 6 rows
    (original rows 0, 1, 2, 3, 4, 7).
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    original_texts = session.dataframe["text"].tolist()
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Check with default parameters (invoke({})) to ensure default drop is False
    res_check = check_duplicates_tool.invoke({})
    assert res_check["status"] == "success"
    assert res_check["n_documents_before"] == 8
    assert res_check["n_duplicates"] == 1
    assert res_check["n_dropped"] == 0
    assert res_check["n_documents_after"] == 8
    assert res_check["dropped"] is False
    assert len(session.dataframe) == 8

    # Check with drop=True
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["n_documents_before"] == 8
    assert res_drop["n_duplicates"] == 1
    assert res_drop["n_dropped"] == 2
    assert res_drop["n_documents_after"] == 6
    assert res_drop["dropped"] is True
    assert len(session.dataframe) == 6

    # Verify remaining rows correspond to original indices [0, 1, 2, 3, 4, 7]
    expected_texts = [original_texts[i] for i in [0, 1, 2, 3, 4, 7]]
    actual_texts = session.dataframe["text"].tolist()
    assert actual_texts == expected_texts


def test_check_duplicates_tool_after_tokenize():
    """
    Test that check_duplicates_tool works properly when session.dataframe contains
    an unhashable unigrams list column created by tokenize_tool.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Run tokenize_tool first
    tokenize_tool.invoke({})
    assert "unigrams" in session.dataframe.columns

    # Run check_duplicates_tool
    res = check_duplicates_tool.invoke({"drop": True})
    assert res["status"] == "success"
    assert res["n_dropped"] == 2
    assert len(session.dataframe) == 6


def test_check_duplicates_tool_dtm_invalidation():
    """
    Test that when check_duplicates_tool drops rows, it invalidates existing DTM and labels in session.
    """
    from tools_dtm import make_tools as make_dtm_tools

    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    assert session.feature_matrix is not None
    assert session.labels is not None

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    check_duplicates_tool.invoke({"drop": True})

    # State must be invalidated to prevent shape mismatch in downstream tools
    assert session.feature_matrix is None
    assert session.labels is None


def test_check_duplicates_tool_no_dataframe_error():
    """
    Test that check_duplicates_tool returns an error status when no dataframe is loaded.
    """
    session = SessionState()
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    result = check_duplicates_tool.invoke({})
    assert result["status"] == "error"
    assert "No dataset loaded" in result["message"]


def test_sample_data_tool_known_answer_13():
    """
    Test sample_data_tool against TEST_FIXTURE.md known answer 13 using sample_fixture.csv.
    Asserts that n=4, random_state=42 returns exactly sampled indices [1, 5, 0, 7] in order,
    and updates session.dataframe to 4 rows.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    original_texts = session.dataframe["text"].tolist()
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert result["status"] == "success"
    assert result["n_documents_before"] == 8
    assert result["n_documents_after"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4

    # Verify session.dataframe rows match original indices [1, 5, 0, 7]
    expected_texts = [original_texts[i] for i in [1, 5, 0, 7]]
    actual_texts = session.dataframe["text"].tolist()
    assert actual_texts == expected_texts


def test_sample_data_tool_errors_and_edge_cases():
    """
    Test error handling for sample_data_tool:
    - n exceeds dataset length when replace=False (dataframe untouched)
    - No n or frac provided (dataframe untouched)
    - No dataset loaded in session
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # 1. n > len(dataframe) when replace=False
    res_exceed = sample_data_tool.invoke({"n": 20, "replace": False})
    assert res_exceed["status"] == "error"
    assert len(session.dataframe) == 8  # Dataframe untouched

    # 2. Call without n or frac
    res_no_param = sample_data_tool.invoke({})
    assert res_no_param["status"] == "error"
    assert len(session.dataframe) == 8  # Dataframe untouched

    # 3. No dataset loaded
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    sample_empty_tool = next(t for t in tools_empty if t.name == "sample_data_tool")
    res_empty = sample_empty_tool.invoke({"n": 4})
    assert res_empty["status"] == "error"
    assert "No dataset loaded" in res_empty["message"]


def test_sample_data_tool_dtm_invalidation():
    """
    Test that sample_data_tool invalidates existing DTM and labels state in session when run.
    """
    from tools_dtm import make_tools as make_dtm_tools

    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    assert session.feature_matrix is not None
    assert session.labels is not None

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    sample_data_tool.invoke({"n": 4, "random_state": 42})

    # DTM state must be invalidated after sampling
    assert session.feature_matrix is None
    assert session.labels is None


def test_describe_data_tool_known_answer_18():
    """
    Test describe_data_tool against TEST_FIXTURE.md known answer 18 using sample_fixture.csv.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})
    assert result["status"] == "success"

    # Verify session.pending_figure is a matplotlib Figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)

    # 1. Overall describe stats
    overall = result["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30.0, abs=1e-3)

    # 2. catA describe stats
    catA = result["by_category"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == pytest.approx(0.0, abs=1e-3)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == pytest.approx(29.0, abs=1e-3)

    # 3. catB describe stats
    catB = result["by_category"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18.0, abs=1e-3)
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == pytest.approx(30.0, abs=1e-3)


def test_describe_data_tool_single_doc_category_json_serializable():
    """
    Test describe_data_tool on a dataset where one category has only 1 document (std becomes NaN/None).
    Ensures that summary is cleanly JSON serializable without throwing errors.
    """
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["hello world", "foo bar", "single doc"],
        "category_name": ["catA", "catA", "catB"]
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})
    assert result["status"] == "success"

    # catB has only 1 row, so std should be None in cleaned dict
    catB_stats = result["by_category"]["catB"]
    assert catB_stats["count"] == 1
    assert catB_stats["std"] is None

    # Test JSON serialization with allow_nan=False
    dumped = json.dumps(result, allow_nan=False)
    assert isinstance(dumped, str)


def test_describe_data_tool_no_dataframe_error():
    """
    Test that describe_data_tool returns an error status when no dataframe is loaded in session.
    """
    session = SessionState()
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})
    assert result["status"] == "error"
    assert "No dataset loaded" in result["message"]


def test_inspect_data_tool_default_params():
    """
    Test inspect_data_tool with default parameters (start=0, n=5) on sample_fixture.csv.
    Asserts exact values from TEST_FIXTURE.md, category integer codes, JSON serializability, and dataframe immutability.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    df_copy_before = session.dataframe.copy()

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({})
    assert result["status"] == "success"
    assert result["total_rows"] == 8
    assert result["n_returned"] == 5

    # Check Row 0
    row_0 = result["rows"][0]
    assert row_0["row_index"] == 0
    assert row_0["text"] == "always alpha alpha alpha beta"
    assert row_0["category_name"] == "catA"
    assert row_0["category"] == 0

    # Check Row 3
    row_3 = result["rows"][3]
    assert row_3["row_index"] == 3
    assert row_3["text"] == "always gamma gamma gamma delta"
    assert row_3["category_name"] == "catB"
    assert row_3["category"] == 1

    # Verify session.dataframe was NOT mutated
    pd.testing.assert_frame_equal(session.dataframe, df_copy_before)

    # Verify JSON serializability
    dumped = json.dumps(result, allow_nan=False)
    assert isinstance(dumped, str)


def test_inspect_data_tool_out_of_bounds_clip():
    """
    Test inspect_data_tool with start=6, n=5 on sample_fixture.csv.
    Should return exactly rows 6 and 7 (2 rows total). Row 7 has empty text with len 0.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({"start": 6, "n": 5})
    assert result["status"] == "success"
    assert result["n_returned"] == 2

    # Row 7 is empty text
    row_7 = result["rows"][1]
    assert row_7["row_index"] == 7
    assert row_7["text"] == ""
    assert len(row_7["text"]) == 0

    # Verify JSON serializability
    dumped = json.dumps(result, allow_nan=False)
    assert isinstance(dumped, str)


def test_inspect_data_tool_invalid_params_error():
    """
    Test inspect_data_tool parameter validation for n and start.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Invalid n values
    for invalid_n in [0, -1, 21]:
        res = inspect_data_tool.invoke({"n": invalid_n})
        assert res["status"] == "error"

    # Invalid start values
    for invalid_start in [-1, 8]:
        res = inspect_data_tool.invoke({"start": invalid_start})
        assert res["status"] == "error"


def test_inspect_data_tool_after_tokenize():
    """
    Test inspect_data_tool after tokenize_tool creates 'unigrams' list column.
    Asserts exact unigrams tokens derived from Known Answer 10 for row 0.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Tokenize first
    tokenize_tool.invoke({})

    result = inspect_data_tool.invoke({"start": 0, "n": 4})
    assert result["status"] == "success"
    assert "unigrams" in result["columns"]

    # Row 0 text: "always alpha alpha alpha beta" -> expected tokens per Known Answer 10
    expected_row_0_tokens = ["always", "alpha", "alpha", "alpha", "beta"]
    actual_row_0_tokens = result["rows"][0]["unigrams"]
    assert actual_row_0_tokens == expected_row_0_tokens

    # Verify JSON serializability
    dumped = json.dumps(result, allow_nan=False)
    assert isinstance(dumped, str)


def test_inspect_data_tool_truncation_long_text_and_unigrams():
    """
    Test that inspect_data_tool correctly truncates text (>200 chars) and unigrams (>20 items),
    matching exact expected truncated string representations.
    """
    session = SessionState()
    # Construct 300-char text and 35-item unigrams list
    long_text = "word " * 60  # 300 characters
    long_unigrams = [f"token_{i}" for i in range(35)]  # 35 tokens

    session.dataframe = pd.DataFrame({
        "text": [long_text],
        "unigrams": [long_unigrams]
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({"start": 0, "n": 1})
    assert result["status"] == "success"

    row = result["rows"][0]

    # Exact string equality assertions for truncation
    expected_text = f"{long_text[:200]}... [truncated, original length: 300 chars]"
    expected_unigrams = f"{long_unigrams[:20]}... [truncated, total items: 35]"

    assert row["text"] == expected_text
    assert row["unigrams"] == expected_unigrams

    # Verify JSON serializability
    dumped = json.dumps(result, allow_nan=False)
    assert isinstance(dumped, str)


def test_inspect_data_tool_no_dataframe_error():
    """
    Test that inspect_data_tool returns an error status when no dataframe is loaded.
    """
    session = SessionState()
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({})
    assert result["status"] == "error"
    assert "No dataset loaded" in result["message"]
