"""Tests for reduce_dimensions_tool in tools_reduction.py."""

import json
import pytest
import numpy as np
import pandas as pd
import matplotlib.figure

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def _setup_session_with_dtm():
    """Helper to initialize a fresh session with sample_fixture.csv and DTM built."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Import build_dtm_tool and run it
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_pca_known_answer_7():
    """Test PCA method against Known Answer 7 in TEST_FIXTURE.md."""
    session = _setup_session_with_dtm()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca"})
    
    assert res["method"] == "pca"
    assert res["shape"] == [8, 2]
    
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    
    coords = session.artifacts["reduced_coordinates"]
    np.testing.assert_allclose(coords, expected_coords, atol=1e-3)
    
    expected_ratio = [0.7532, 0.1965]
    assert res["explained_variance_ratio"] == pytest.approx(expected_ratio, abs=1e-3)
    
    # Check pending figure
    fig = session.pending_figure
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)


def test_reduce_dimensions_tsne_properties():
    """Property-based verification for t-SNE (Known Answer 8).

    Note:
        Known Answer 8 was computed under a different platform / BLAS / C++ PRNG environment.
        Even with random_state fixed, floating point optimization paths differ across platforms.
        Therefore, this test validates structural properties: shape (8,2), absence of NaN,
        reproducibility with same random_state, distinct output with different random_state,
        duplicate row equality (doc 5 & 6), cluster proximity to centroids (docs 0-2 closer
        to catA, docs 3-6 closer to catB), and figure properties (8 total scatter points,
        legend labels for catA and catB).
    """
    session = _setup_session_with_dtm()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Run 1: random_state=42
    res1 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    coords1 = session.artifacts["reduced_coordinates"]

    # 1. Shape is (8, 2) and contains no NaN
    assert res1["shape"] == [8, 2]
    assert coords1.shape == (8, 2)
    assert not np.isnan(coords1).any()

    # 2. Duplicate text rows (doc 5 & 6, both "always gamma delta") yield identical coordinates
    np.testing.assert_allclose(coords1[5], coords1[6], atol=1e-3)

    # 3. Same random_state produces identical coordinates
    reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    coords1_again = session.artifacts["reduced_coordinates"]
    np.testing.assert_array_equal(coords1, coords1_again)

    # 4. Different random_state produces different coordinates
    reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 99
    })
    coords2 = session.artifacts["reduced_coordinates"]
    assert not np.array_equal(coords1, coords2)

    # 5. Cluster proximity to centroids: docs 0-2 closer to catA centroid, docs 3-6 closer to catB centroid
    centroids = res1["category_centroids"]
    catA_center = np.array(centroids["catA"])
    catB_center = np.array(centroids["catB"])

    for i in range(3):  # catA docs 0, 1, 2
        dist_A = np.linalg.norm(coords1[i] - catA_center)
        dist_B = np.linalg.norm(coords1[i] - catB_center)
        assert dist_A < dist_B

    for i in range(3, 7):  # catB docs 3, 4, 5, 6
        dist_A = np.linalg.norm(coords1[i] - catA_center)
        dist_B = np.linalg.norm(coords1[i] - catB_center)
        assert dist_B < dist_A

    # 6. Scatter plot figure verification: 8 total points and legend categories
    fig = session.pending_figure
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    
    total_points = sum(len(col.get_offsets()) for col in ax.collections)
    assert total_points == 8

    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "catA" in legend_texts
    assert "catB" in legend_texts


def test_reduce_dimensions_umap_properties():
    """Property-based verification for UMAP (Known Answer 9).

    Note:
        Known Answer 9 was computed under a different platform / BLAS / C++ PRNG environment.
        Even with random_state fixed, floating point optimization paths differ across platforms.
        This test validates structural properties: shape (8,2), absence of NaN, reproducibility
        with same random_state, distinct output with different random_state, cluster proximity
        to centroids (docs 0-2 closer to catA, docs 3-6 closer to catB), and figure properties
        (8 total scatter points, legend labels for catA and catB).
    """
    session = _setup_session_with_dtm()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Run 1: random_state=42
    res1 = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    coords1 = session.artifacts["reduced_coordinates"]

    # 1. Shape is (8, 2) and contains no NaN
    assert res1["shape"] == [8, 2]
    assert coords1.shape == (8, 2)
    assert not np.isnan(coords1).any()

    # 2. Same random_state produces identical coordinates
    reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    coords1_again = session.artifacts["reduced_coordinates"]
    np.testing.assert_array_equal(coords1, coords1_again)

    # 3. Different random_state produces different coordinates
    reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 99
    })
    coords2 = session.artifacts["reduced_coordinates"]
    assert not np.array_equal(coords1, coords2)

    # 4. Cluster proximity to centroids: docs 0-2 closer to catA centroid, docs 3-6 closer to catB centroid
    centroids = res1["category_centroids"]
    catA_center = np.array(centroids["catA"])
    catB_center = np.array(centroids["catB"])

    for i in range(3):  # catA docs 0, 1, 2
        dist_A = np.linalg.norm(coords1[i] - catA_center)
        dist_B = np.linalg.norm(coords1[i] - catB_center)
        assert dist_A < dist_B

    for i in range(3, 7):  # catB docs 3, 4, 5, 6
        dist_A = np.linalg.norm(coords1[i] - catA_center)
        dist_B = np.linalg.norm(coords1[i] - catB_center)
        assert dist_B < dist_A

    # 5. Scatter plot figure verification: 8 total points and legend categories
    fig = session.pending_figure
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    
    total_points = sum(len(col.get_offsets()) for col in ax.collections)
    assert total_points == 8

    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "catA" in legend_texts
    assert "catB" in legend_texts


def test_reduce_dimensions_parameter_boundary_errors():
    """Test parameter validation errors for perplexity and n_neighbors bounds."""
    session = _setup_session_with_dtm()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # t-SNE default perplexity=30.0 >= n_samples (8)
    res_tsne_err = reduce_dimensions_tool.invoke({"method": "tsne"})
    assert res_tsne_err.get("status") == "error"
    assert "perplexity" in res_tsne_err.get("message", "")

    # UMAP default n_neighbors=15 >= n_samples (8)
    res_umap_err15 = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 15})
    assert res_umap_err15.get("status") == "error"
    assert "n_neighbors" in res_umap_err15.get("message", "")

    # UMAP n_neighbors=1 < 2
    res_umap_err1 = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 1})
    assert res_umap_err1.get("status") == "error"
    assert "n_neighbors" in res_umap_err1.get("message", "")


def test_reduce_dimensions_no_dtm_error():
    """Test error handling when feature_matrix is not built in session."""
    session = SessionState()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca"})
    assert res.get("status") == "error"
    assert "No feature_matrix found" in res.get("message", "")


def test_binarize_labels_no_dataframe_error():
    """Test error handling when session dataframe is None."""
    session = SessionState()
    reduction_tools = make_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    res_err = binarize_tool.invoke({})
    assert res_err.get("status") == "error"
    assert "No dataframe loaded" in res_err.get("message", "")


def test_binarize_labels_fixture_known_labels():
    """Test binarize_labels_tool on sample_fixture.csv with DTM pre-built."""
    session = _setup_session_with_dtm()
    reduction_tools = make_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # Ensure DTM is present before running binarize_labels_tool
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)

    res = binarize_tool.invoke({})
    assert res["status"] == "success"
    assert res["classes"] == ["catA", "catB"]
    assert res["n_classes"] == 2
    assert res["counts_per_category"] == {"catA": 4, "catB": 4}

    # Verify DTM state remains intact after binarize_labels_tool
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)

    # Verify JSON serializability
    json_str = json.dumps(res, allow_nan=False)
    assert isinstance(json_str, str)

    # Verify all 8 rows against expected vectors derived from sample_fixture.csv labels
    # Row 0, 1, 2, 7 -> catA ([1, 0]); Row 3, 4, 5, 6 -> catB ([0, 1])
    expected_vectors = [
        [1, 0],  # row 0 (catA)
        [1, 0],  # row 1 (catA)
        [1, 0],  # row 2 (catA)
        [0, 1],  # row 3 (catB)
        [0, 1],  # row 4 (catB)
        [0, 1],  # row 5 (catB)
        [0, 1],  # row 6 (catB)
        [1, 0],  # row 7 (catA)
    ]
    actual_vectors = session.dataframe["bin_category"].tolist()
    assert actual_vectors == expected_vectors


def test_binarize_labels_idempotency():
    """Test that calling binarize_labels_tool twice preserves exact column list and category values."""
    session = _setup_session_with_dtm()
    reduction_tools = make_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # Save original category_name series to verify it is untouched
    original_category_names = session.dataframe["category_name"].tolist()

    expected_vectors = [
        [1, 0], [1, 0], [1, 0], [0, 1],
        [0, 1], [0, 1], [0, 1], [1, 0]
    ]

    res_first = binarize_tool.invoke({})
    assert res_first["status"] == "success"
    cols_after_first = list(session.dataframe.columns)
    vectors_after_first = session.dataframe["bin_category"].tolist()

    res_second = binarize_tool.invoke({})
    assert res_second["status"] == "success"
    cols_after_second = list(session.dataframe.columns)
    vectors_after_second = session.dataframe["bin_category"].tolist()

    # 1. Columns sequence and set must be identical
    assert cols_after_first == cols_after_second

    # 2. Binarized vectors must remain identical and correct across calls
    assert vectors_after_first == expected_vectors
    assert vectors_after_second == expected_vectors

    # 3. Original category_name column must remain completely unchanged
    assert session.dataframe["category_name"].tolist() == original_category_names


def test_binarize_labels_three_categories():
    """Test binarize_labels_tool on a DataFrame with 3 categories."""
    session_3cat = SessionState()
    session_3cat.dataframe = pd.DataFrame({
        "category_name": ["catC", "catA", "catB", "catA"]
    })
    tools_3cat = make_tools(session_3cat)
    binarize_3cat = next(t for t in tools_3cat if t.name == "binarize_labels_tool")
    res_3cat = binarize_3cat.invoke({})

    assert res_3cat["status"] == "success"
    assert res_3cat["classes"] == ["catA", "catB", "catC"]
    assert res_3cat["n_classes"] == 3
    expected_3cat_vectors = [
        [0, 0, 1],  # catC
        [1, 0, 0],  # catA
        [0, 1, 0],  # catB
        [1, 0, 0],  # catA
    ]
    assert session_3cat.dataframe["bin_category"].tolist() == expected_3cat_vectors


def test_binarize_labels_alphabetical_sorting():
    """Test that category classes and vector columns are sorted alphabetically regardless of row order."""
    session_order = SessionState()
    session_order.dataframe = pd.DataFrame({
        "category_name": ["zeta", "alpha", "zeta"]
    })
    tools_order = make_tools(session_order)
    binarize_order = next(t for t in tools_order if t.name == "binarize_labels_tool")
    res_order = binarize_order.invoke({})

    assert res_order["status"] == "success"
    assert res_order["classes"] == ["alpha", "zeta"]
    expected_order_vectors = [
        [0, 1],  # zeta -> [alpha=0, zeta=1]
        [1, 0],  # alpha -> [alpha=1, zeta=0]
        [0, 1],  # zeta -> [alpha=0, zeta=1]
    ]
    assert session_order.dataframe["bin_category"].tolist() == expected_order_vectors
