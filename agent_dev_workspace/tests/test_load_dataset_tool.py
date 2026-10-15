import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # 建立測試用的 SessionState 實例
    session = SessionState()
    
    # 載入預置工具 (premade_tools 中包含 load_dataset_tool)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 執行 load_dataset_tool 讀取 sample_fixture.csv
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 驗證已知答案 (TEST_FIXTURE.md #1)
    # 1. 總文件數為 8
    assert result["n_documents"] == 8
    
    # 2. 類別統計：catA: 4, catB: 4
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
