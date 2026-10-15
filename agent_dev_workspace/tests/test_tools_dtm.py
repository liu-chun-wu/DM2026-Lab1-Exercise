import json
import pytest
import numpy as np
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # 建立 shared session
    session = SessionState()

    # 載入預置工具 (load_dataset_tool)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 1. 載入範例資料集 sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. 載入並執行 build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    result = build_dtm_tool.invoke({})

    # 3. 對照 TEST_FIXTURE.md Known Answer #4 進行斷言檢查
    # 詞彙表：5 個詞彙 (alpha, always, beta, delta, gamma)
    assert result["n_terms"] == 5
    assert result["vocabulary_sample"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # 矩陣結構與非零值：8 列 5 欄，共 21 個非零格（out of 40）
    assert result["n_documents"] == 8
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40

    # 稀疏度：100 * (1 - 21/40) = 47.5%
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)

    # 檢查 artifact 是否存有 count_vectorizer
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None


def test_term_frequency_tool_known_answer_14():
    # 建立獨立的新 session
    session = SessionState()

    # 載入 load_dataset_tool 並載入資料
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 執行 build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})

    # 執行 term_frequency_tool
    result = term_frequency_tool.invoke({})

    # 1. 檢查 summary 可被 JSON 序列化
    serialized = json.dumps(result)
    assert serialized is not None

    # 2. 對照 Known Answer 14 檢查各詞總數
    expected_counts = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    assert result["total_terms"] == 5
    assert result["total_word_count"] == 27  # 6 + 7 + 3 + 4 + 7
    for term, count in expected_counts.items():
        assert result["top_terms"][term] == count

    # 3. 檢查 full_report 前兩欄名稱與順序 (第一欄為 term，第二欄為 frequency)
    result_id = result["result_id"]
    stored = session.results_store[result_id]
    full_report = stored["full_report"]

    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "frequency"

    # 4. 檢查完整排序順序 (frequency 降序，term 升序)
    # always (7) 與 gamma (7) 並列，"always" < "gamma"，故 always 在前
    expected_order = ["always", "gamma", "alpha", "delta", "beta"]
    actual_order = full_report["term"].tolist()
    assert actual_order == expected_order


def test_term_frequency_tool_errors():
    # 邊界測試 1: 未建立 DTM 即呼叫
    session1 = SessionState()
    dtm_tools1 = make_tools(session1)
    term_frequency_tool1 = next(t for t in dtm_tools1 if t.name == "term_frequency_tool")

    res1 = term_frequency_tool1.invoke({})
    assert res1["status"] == "error"
    assert "not found" in res1["message"]

    # 邊界測試 2: top_k <= 0
    session2 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    dtm_tools2 = make_tools(session2)
    build_dtm_tool2 = next(t for t in dtm_tools2 if t.name == "build_dtm_tool")
    term_frequency_tool2 = next(t for t in dtm_tools2 if t.name == "term_frequency_tool")

    build_dtm_tool2.invoke({})
    res2 = term_frequency_tool2.invoke({"top_k": 0})
    assert res2["status"] == "error"
    assert "positive integer" in res2["message"]


def test_dtm_heatmap_tool_known_answer_21():
    # 建立獨立的新 session
    session = SessionState()

    # 載入 load_dataset_tool 並載入 sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 取得工具組並建立 DTM
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # 1. 預設參數 (n_terms=20, n_documents=20) 呼叫，測試完整 8x5 矩陣與 Known Answer 21
    res = dtm_heatmap_tool.invoke({})

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    # 比對完整的 8x5 矩陣數據 (逐格精確比對)
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res["matrix_slice"] == expected_full_matrix

    # 驗證 session.pending_figure 為 matplotlib Figure 實例
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # 2. 測試自訂切片參數 (n_terms=2, n_documents=3)
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_slice["status"] == "success"
    assert res_slice["n_documents"] == 3
    assert res_slice["n_terms"] == 2
    assert res_slice["terms"] == ["alpha", "always"]
    assert res_slice["matrix_slice"] == [
        [3, 1],
        [2, 1],
        [1, 1]
    ]


def test_dtm_heatmap_tool_errors():
    # 邊界測試 1: 未建立 DTM 即呼叫
    session1 = SessionState()
    dtm_tools1 = make_tools(session1)
    dtm_heatmap_tool1 = next(t for t in dtm_tools1 if t.name == "dtm_heatmap_tool")

    res1 = dtm_heatmap_tool1.invoke({})
    assert res1["status"] == "error"
    assert "not found" in res1["message"]

    # 邊界測試 2: n_terms <= 0 或 n_documents <= 0
    session2 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    dtm_tools2 = make_tools(session2)
    build_dtm_tool2 = next(t for t in dtm_tools2 if t.name == "build_dtm_tool")
    dtm_heatmap_tool2 = next(t for t in dtm_tools2 if t.name == "dtm_heatmap_tool")

    build_dtm_tool2.invoke({})

    # n_terms = -1
    res2 = dtm_heatmap_tool2.invoke({"n_terms": -1})
    assert res2["status"] == "error"
    assert "n_terms must be a positive integer" in res2["message"]

    # n_documents = 0
    res3 = dtm_heatmap_tool2.invoke({"n_documents": 0})
    assert res3["status"] == "error"
    assert "n_documents must be a positive integer" in res3["message"]
