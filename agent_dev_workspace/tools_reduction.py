"""Dimensionality reduction tool for document-term matrix (tools_reduction.py)."""

from typing import Optional, Literal, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap
from threadpoolctl import threadpool_limits

from langchain_core.tools import tool


def make_tools(session):
    """Factory function creating tools for dimensionality reduction bound to session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reduces the dimensionality of session.feature_matrix to 2D coordinates and plots a scatter plot.

        Supports PCA, t-SNE, and UMAP methods. The resulting coordinates are saved to
        session.artifacts and stored in session results. A scatter plot colored by category
        is automatically created and set on session.pending_figure.

        Note on Reproducibility:
            When random_state is None, t-SNE and UMAP yield non-deterministic results due to
            stochastic initialization and optimization. Pass an explicit integer random_state
            for reproducible output.

        Args:
            method: Reduction algorithm to use ("pca", "tsne", or "umap"). Defaults to "pca".
            perplexity: Perplexity parameter for t-SNE (used when method="tsne"). Must be < n_samples. Defaults to 30.0.
            n_neighbors: Number of neighbors parameter for UMAP (used when method="umap"). Must be in [2, n_samples - 1]. Defaults to 15.
            random_state: Optional random seed for reproducible reduction.

        Returns:
            Dict containing result_id, method, shape, params, explained_variance_ratio (for PCA),
            and category_centroids.
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "message": "No feature_matrix found in session. Please run build_dtm_tool first.",
            }

        X = session.feature_matrix
        n_samples = X.shape[0]
        if n_samples == 0:
            return {
                "status": "error",
                "message": "feature_matrix is empty.",
            }

        explained_variance_ratio: Optional[List[float]] = None
        params_used: Dict[str, Any] = {
            "method": method,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state,
        }

        if method == "pca":
            model = PCA(n_components=2, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]

        elif method == "tsne":
            if perplexity >= n_samples or perplexity <= 0:
                return {
                    "status": "error",
                    "message": f"perplexity ({perplexity}) must be strictly positive and less than n_samples ({n_samples}).",
                }

            # For high-dimensional sparse inputs, pre-reduce with TruncatedSVD to 50 dims if n_features > 50
            if hasattr(X, "toarray"):
                if X.shape[1] > 50:
                    X_input = TruncatedSVD(n_components=50, random_state=random_state).fit_transform(X)
                else:
                    X_input = X.toarray()
            else:
                X_input = np.asarray(X)

            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
            )
            # Limit OpenMP threads to 4 using threadpoolctl to prevent severe thread-thrashing overhead
            # in multi-core CPU environments where n_jobs on TSNE has no effect.
            with threadpool_limits(limits=4, user_api="openmp"):
                coords = model.fit_transform(X_input)

        elif method == "umap":
            if n_neighbors >= n_samples or n_neighbors < 2:
                return {
                    "status": "error",
                    "message": f"n_neighbors ({n_neighbors}) must be between 2 and n_samples - 1 ({n_samples - 1}).",
                }

            # UMAP natively handles sparse matrices directly
            model = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = model.fit_transform(X)

        else:
            return {
                "status": "error",
                "message": f"Unsupported method: {method}. Choose from 'pca', 'tsne', 'umap'.",
            }

        # Retrieve category labels
        labels = None
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].values
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].values

        # Compute category centroids
        category_centroids: Dict[str, List[float]] = {}
        if labels is not None and len(labels) == n_samples:
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                mask = (labels == lbl)
                centroid = coords[mask].mean(axis=0).tolist()
                category_centroids[str(lbl)] = [round(float(c), 4) for c in centroid]

        # Generate scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None and len(labels) == n_samples:
            df_plot = pd.DataFrame(
                {
                    "Dim 1": coords[:, 0],
                    "Dim 2": coords[:, 1],
                    "Category": labels,
                }
            )
            sns.scatterplot(
                data=df_plot,
                x="Dim 1",
                y="Dim 2",
                hue="Category",
                style="Category",
                ax=ax,
                s=100,
            )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.8, s=100)

        ax.set_title(f"{method.upper()} Dimensionality Reduction")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        fig.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Store full coordinates in artifacts
        session.artifacts["reduced_coordinates"] = coords
        session.artifacts[f"reduced_{method}"] = coords

        result_id = session.next_result_id("reduce_dims")
        summary = {
            "status": "success",
            "result_id": result_id,
            "method": method,
            "shape": list(coords.shape),
            "params": params_used,
            "explained_variance_ratio": explained_variance_ratio,
            "category_centroids": category_centroids,
        }

        session.store_result("reduce_dimensions_tool", params_used, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes document category labels into one-hot vectors and stores them in session.dataframe.

        Converts string category labels from session.dataframe['category_name'] into one-hot binary
        vectors aligned with sorted category names (matching session.categories ordering). The resulting
        vector list for each document is stored as a new column 'bin_category' in session.dataframe.

        Returns:
            Dict containing status, result_id, classes (ordered category names corresponding to vector positions),
            n_classes, counts_per_category, and a preview of the first 5 rows.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Please run load_dataset_tool first.",
            }

        if "category_name" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Column 'category_name' not found in session.dataframe.",
            }

        cat_series = session.dataframe["category_name"]
        unique_cats = sorted(cat_series.dropna().unique().tolist())

        if not unique_cats:
            return {
                "status": "error",
                "message": "No valid categories found in session.dataframe['category_name'].",
            }

        # Construct one-hot vectors ordered by sorted category names
        # e.g., cat -> [1, 0, ...] where index matches position in unique_cats
        cat_to_vector = {
            cat: [1 if i == idx else 0 for i in range(len(unique_cats))]
            for idx, cat in enumerate(unique_cats)
        }

        # Apply mapping to dataframe column bin_category using .copy() to ensure each row gets an independent list object
        session.dataframe["bin_category"] = cat_series.map(
            lambda c: cat_to_vector[c].copy() if c in cat_to_vector else [0] * len(unique_cats)
        )

        counts = cat_series.value_counts().to_dict()
        preview = session.dataframe["bin_category"].head(5).tolist()

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "status": "success",
            "result_id": result_id,
            "classes": unique_cats,
            "n_classes": len(unique_cats),
            "counts_per_category": counts,
            "preview": preview,
        }

        session.store_result("binarize_labels_tool", {}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
