from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Builds a document-term matrix from session.dataframe text and reports its sparsity.

        Args:
            ngram_range: Optional list of two ints [min_n, max_n] specifying the range of n-gram values. Defaults to [1, 1].
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataframe or 'text' column found in session."}

        # Handle default value and validate length for ngram_range
        if ngram_range is None:
            ngram_tuple = (1, 1)
        else:
            if not isinstance(ngram_range, list) or len(ngram_range) != 2:
                return {
                    "status": "error",
                    "message": f"ngram_range must be a list of exactly 2 integers, got {ngram_range}",
                }
            ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
        )

        texts = session.dataframe["text"].fillna("").astype(str)
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        session.artifacts["count_vectorizer"] = vectorizer

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (
            100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0
        )

        result_id = session.next_result_id("dtm")
        # Keep summary small for LLM prompt context: preview top 10 terms
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "vocabulary_sample": feature_names[:10],
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
        }

        # Full vocabulary goes into full_report (not sent directly to LLM)
        full_report = {
            "vocabulary": feature_names,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "sparsity_pct": sparsity_pct,
        }

        session.store_result(
            "build_dtm_tool",
            {"ngram_range": list(ngram_tuple)},
            summary,
            full_report=full_report,
        )
        return summary

    @tool
    def term_frequency_tool(
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the Document-Term Matrix.

        Args:
            top_k: Number of highest frequency terms to include in the summary preview (between 1 and 100). Defaults to 20.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Feature matrix or feature names not found. Please run build_dtm_tool first.",
            }

        if top_k <= 0:
            return {
                "status": "error",
                "message": f"top_k must be a positive integer, got {top_k}.",
            }

        if top_k > 100:
            top_k = 100

        # Compute column sums directly on sparse matrix for memory efficiency
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()

        df = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts,
        })

        # Sort by frequency descending, then by term ascending for stable tie-breaking
        df = df.sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        result_id = session.next_result_id("term_freq")

        top_df = df.head(top_k)
        top_terms = {row["term"]: int(row["frequency"]) for _, row in top_df.iterrows()}

        summary = {
            "result_id": result_id,
            "total_terms": int(len(df)),
            "total_word_count": int(df["frequency"].sum()),
            "top_terms": top_terms,
        }

        session.store_result(
            "term_frequency_tool",
            {"top_k": top_k},
            summary,
            full_report=df,
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Renders a heatmap of a raw positional slice of the global document-term matrix.

        Args:
            n_terms: How many terms (columns) to include in existing vocabulary order (between 1 and 50, default 20, capped at 50).
            n_documents: How many documents (rows) to include in existing row order (between 1 and 50, default 20, capped at 50).

        Returns:
            Dict containing result_id, n_terms, n_documents, selected terms, document labels, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Feature matrix or feature names not found. Please run build_dtm_tool first.",
            }

        if n_terms <= 0:
            return {
                "status": "error",
                "message": f"n_terms must be a positive integer >= 1, got {n_terms}.",
            }

        if n_documents <= 0:
            return {
                "status": "error",
                "message": f"n_documents must be a positive integer >= 1, got {n_documents}.",
            }

        # Cap maximum dimensions at 50
        if n_terms > 50:
            n_terms = 50

        if n_documents > 50:
            n_documents = 50

        total_docs, total_terms = session.feature_matrix.shape

        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        # Slice sparse matrix BEFORE converting to dense array for memory efficiency
        sliced_sparse = session.feature_matrix[:actual_docs, :actual_terms]
        matrix_dense = sliced_sparse.toarray().tolist()

        selected_terms = session.feature_names[:actual_terms]
        selected_doc_ids = list(range(actual_docs))

        # Create plot matching describe_data_tool pattern
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            sliced_sparse.toarray(),
            xticklabels=selected_terms,
            yticklabels=selected_doc_ids,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax,
        )
        ax.set_title(f"DTM Slice Heatmap ({actual_docs} docs x {actual_terms} terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")

        fig.tight_layout()
        plt.close(fig)
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": selected_terms,
            "doc_indices": selected_doc_ids,
            "matrix_slice": matrix_dense,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
