import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def feature_correlation_matrix_tool(top_k: int = 20) -> dict:
        """Computes a Pearson correlation matrix for the top terms by variance in the global DTM and builds a heatmap.

        Args:
            top_k (int, optional): The maximum number of top-variance terms to include in the matrix (must be between 1 and 30). Defaults to 20.

        Returns:
            dict: A summary containing the result_id, n_features, top_features, and the correlation_matrix (as a nested list).
        """
        if top_k < 1 or top_k > 30:
            return {
                "status": "error",
                "message": f"top_k must be between 1 and 30, got {top_k}.",
            }

        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No feature matrix found in session. Please run build_dtm_tool first.",
            }

        # 1. Compute per-term variance using population variance (ddof=0) without dense conversion
        feature_matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)

        mean = feature_matrix.mean(axis=0)
        mean_sq = feature_matrix.power(2).mean(axis=0)
        variances = np.array(mean_sq - np.square(mean)).ravel()

        # 2. Select top_k terms by variance
        k = min(top_k, len(feature_names))
        if k == 0:
            return {
                "status": "error",
                "message": "Feature matrix has no columns/terms.",
            }

        top_indices = np.argsort(variances)[::-1][:k]
        top_features = feature_names[top_indices].tolist()

        # 3. Slice top_k columns and compute Pearson correlation matrix
        top_dense = feature_matrix[:, top_indices].toarray()
        corr_matrix = np.corrcoef(top_dense, rowvar=False)

        # Handle edge cases (e.g., single feature or NaN from zero variance)
        if k == 1:
            corr_matrix = np.array([[1.0]])
        else:
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Convert to JSON-serializable list rounded to 4 decimal places
        corr_list = [
            [round(float(val), 4) for val in row] for row in corr_matrix
        ]

        # 4. Build correlation heatmap
        fig, ax = plt.subplots(figsize=(max(6, k * 0.8), max(5, k * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_features,
            yticklabels=top_features,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {k} Terms by Variance)")
        plt.tight_layout()

        # Set pending_figure for agentic UI rendering
        session.pending_figure = fig
        plt.close(fig)

        # 5. Store and return result
        res_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": res_id,
            "n_features": k,
            "top_features": top_features,
            "correlation_matrix": corr_list,
        }

        session.store_result(
            "feature_correlation_matrix_tool", {"top_k": top_k}, summary
        )
        return summary

    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> dict:
        """Computes cosine similarity between two documents using raw term frequencies from the global DTM.

        Args:
            doc_index_1 (int): The 0-based row index of the first document (0 to total_documents - 1).
            doc_index_2 (int): The 0-based row index of the second document (0 to total_documents - 1).

        Returns:
            dict: Summary containing result_id, doc_index_1, doc_index_2, similarity score (rounded to 4 decimals), and text previews.
        """
        if session.feature_matrix is None or session.dataframe is None:
            return {
                "status": "error",
                "message": "Feature matrix or dataframe not found in session. Please run build_dtm_tool first.",
            }

        if "count_vectorizer" not in session.artifacts:
            return {
                "status": "error",
                "message": "count_vectorizer not found in session.artifacts. Please run build_dtm_tool first.",
            }

        total_docs = session.feature_matrix.shape[0]

        if not (0 <= doc_index_1 < total_docs):
            return {
                "status": "error",
                "message": f"doc_index_1 must be between 0 and {total_docs - 1}, got {doc_index_1}.",
            }

        if not (0 <= doc_index_2 < total_docs):
            return {
                "status": "error",
                "message": f"doc_index_2 must be between 0 and {total_docs - 1}, got {doc_index_2}.",
            }

        # Extract rows from sparse matrix
        v1 = session.feature_matrix[doc_index_1]
        v2 = session.feature_matrix[doc_index_2]

        # Compute dot product and L2 norms using raw count vectors
        v1_arr = v1.toarray().ravel()
        v2_arr = v2.toarray().ravel()

        norm1 = np.linalg.norm(v1_arr)
        norm2 = np.linalg.norm(v2_arr)

        # Edge case: empty document (or vector with all zero counts) returns 0.0 similarity
        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
        else:
            dot_product = np.dot(v1_arr, v2_arr)
            similarity = round(float(dot_product / (norm1 * norm2)), 4)

        # Get text previews from dataframe for display
        text1 = str(session.dataframe.iloc[doc_index_1].get("text", ""))
        text2 = str(session.dataframe.iloc[doc_index_2].get("text", ""))

        preview1 = text1[:50] + "..." if len(text1) > 50 else text1
        preview2 = text2[:50] + "..." if len(text2) > 50 else text2

        res_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": res_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": similarity,
            "doc_1_preview": preview1,
            "doc_2_preview": preview2,
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary,
        )
        return summary

    return [feature_correlation_matrix_tool, cosine_similarity_tool]
