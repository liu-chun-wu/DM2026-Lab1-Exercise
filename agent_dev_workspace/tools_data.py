import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.tools import tool

# Ensure punkt is available for nltk tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session) -> list:
    """
    Factory function to create data tools bound to the given session.
    """

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """
        Tokenizes document text in session.dataframe into unigrams using NLTK word_tokenize.
        
        Text is lowercased before tokenization to match standard pipeline conventions.
        Stores the resulting token lists in session.dataframe['unigrams'].
        
        Returns:
            Dict containing execution status, result_id, document count, and summary statistics.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "session.dataframe does not contain a 'text' column."
            }

        def _tokenize_text(text: Any) -> list:
            if pd.isna(text) or not text:
                return []
            return nltk.word_tokenize(str(text).lower())

        # Apply tokenization row by row
        unigrams_list = session.dataframe["text"].apply(_tokenize_text).tolist()
        session.dataframe["unigrams"] = unigrams_list

        n_documents = len(unigrams_list)
        total_tokens = sum(len(tokens) for tokens in unigrams_list)
        avg_tokens = round(total_tokens / n_documents, 2) if n_documents > 0 else 0.0

        sample_tokens = []
        for tokens in unigrams_list:
            if tokens:
                sample_tokens = tokens[:10]
                break

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
            "sample_tokens": sample_tokens
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """
        Lists files and subdirectories within a given subdirectory relative to the repository root.
        
        Prevents path traversal outside the repository root.
        
        Args:
            subdirectory: Relative path from repository root (default: "" for repo root).
            
        Returns:
            Dict containing result_id, subdirectory, sorted lists of directories and files, or error status.
        """
        # tools_data.py is located inside the workspace folder, so parent.parent is repo_root
        repo_root = Path(__file__).resolve().parent.parent
        
        # Strip leading slashes to prevent absolute path override
        clean_sub = Path(subdirectory.lstrip("/")) if subdirectory.startswith("/") else Path(subdirectory)
        target_path = (repo_root / clean_sub).resolve()

        # Security check: Ensure target_path is within repo_root
        if not target_path.is_relative_to(repo_root):
            return {
                "status": "error",
                "message": f"Access denied: Path '{subdirectory}' escapes repository root."
            }

        if not target_path.exists() or not target_path.is_dir():
            return {
                "status": "error",
                "message": f"Directory '{subdirectory}' does not exist or is not a directory."
            }

        directories = sorted([p.name for p in target_path.iterdir() if p.is_dir()])
        files = sorted([p.name for p in target_path.iterdir() if p.is_file()])

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    @tool
    def inspect_data_tool(start: int = 0, n: int = 5) -> Dict[str, Any]:
        """
        Inspects a specific range of rows in session.dataframe without mutating it.

        Allows peeking at document contents and columns starting at index `start` for `n` rows.
        Text columns are truncated to 200 characters with original length provided.
        List columns (such as 'unigrams') are truncated to 20 items with total item count provided.
        Does not mutate session.dataframe or affect downstream state.

        Args:
            start: Zero-based row index to start inspecting from (default: 0).
            n: Number of rows to inspect (default: 5, max: 20).

        Returns:
            Dict containing result_id, status, total_rows, columns, start, n_requested,
            n_returned, and rows preview.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        total_rows = len(session.dataframe)
        if total_rows == 0:
            return {
                "status": "error",
                "message": "session.dataframe is empty. Nothing to inspect."
            }

        if start < 0 or start >= total_rows:
            return {
                "status": "error",
                "message": f"Parameter 'start' ({start}) is out of bounds for dataset with {total_rows} rows (valid range: 0 to {total_rows - 1})."
            }

        if n <= 0 or n > 20:
            return {
                "status": "error",
                "message": f"Parameter 'n' must be between 1 and 20, got {n}."
            }

        end = min(start + n, total_rows)
        slice_df = session.dataframe.iloc[start:end]

        rows_data = []
        for row_idx, (_, row) in enumerate(slice_df.iterrows(), start=start):
            row_dict = {"row_index": row_idx}
            for col in session.dataframe.columns:
                val = row[col]
                if val is None:
                    row_dict[col] = None
                elif isinstance(val, (list, tuple, np.ndarray)):
                    val_list = list(val)
                    if len(val_list) > 20:
                        row_dict[col] = f"{val_list[:20]}... [truncated, total items: {len(val_list)}]"
                    else:
                        row_dict[col] = val_list
                elif isinstance(val, str):
                    if len(val) > 200:
                        row_dict[col] = f"{val[:200]}... [truncated, original length: {len(val)} chars]"
                    else:
                        row_dict[col] = val
                elif pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, int)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    row_dict[col] = float(val)
                else:
                    row_dict[col] = str(val)
            rows_data.append(row_dict)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "columns": list(session.dataframe.columns),
            "start": start,
            "n_requested": n,
            "n_returned": len(rows_data),
            "rows": rows_data
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"start": start, "n": n},
            summary=summary
        )

        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """
        Checks session.dataframe for missing or empty document text entries.
        
        Identifies rows where 'text' is null/NaN or consists purely of whitespace.
        Does not mutate session.dataframe.
        
        Returns:
            Dict containing result_id, total documents, number of missing documents,
            missing percentage, and a sample preview of missing document indices.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "session.dataframe does not contain a 'text' column."
            }

        text_series = session.dataframe["text"]
        
        # Identify missing entries: NaN/null OR empty string (including whitespace-only)
        is_na = text_series.isna()
        is_empty = text_series.fillna("").astype(str).str.strip() == ""
        missing_mask = is_na | is_empty

        missing_indices = session.dataframe.index[missing_mask].tolist()
        n_missing = len(missing_indices)
        n_documents = len(session.dataframe)
        missing_pct = round((n_missing / n_documents) * 100, 2) if n_documents > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_documents,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "missing_indices": missing_indices[:10]  # Return preview of up to 10 indices
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """
        Checks for (and optionally drops) duplicate documents in session.dataframe.
        
        Evaluates duplicates specifically on the 'text' column (ignoring unhashable columns
        such as 'unigrams').
        
        When drop=True, removes every copy of a duplicated row using keep=False,
        matching Master's X.drop_duplicates(keep=False, inplace=True) behavior.
        If rows are dropped, resets index and invalidates downstream DTM state (feature_matrix,
        feature_names, labels) to prevent shape mismatches.
        
        Args:
            drop: If True, drops all duplicated rows (keep=False) and updates session.dataframe.
            
        Returns:
            Dict containing result_id, status, initial document count, duplicate count
            (calculated with keep='first'), dropped rows count, and remaining document count.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "session.dataframe does not contain a 'text' column."
            }

        n_documents_before = len(session.dataframe)
        
        # Evaluate duplicates specifically on the 'text' column to prevent unhashable type errors
        # (e.g. if 'unigrams' column containing list exists)
        dup_mask_first = session.dataframe.duplicated(subset=["text"], keep="first")
        n_duplicates = int(dup_mask_first.sum())

        n_dropped = 0
        if drop and n_duplicates > 0:
            # Use keep=False when dropping, matching Master's X.drop_duplicates(keep=False, inplace=True)
            dup_mask_all = session.dataframe.duplicated(subset=["text"], keep=False)
            n_dropped = int(dup_mask_all.sum())
            
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

            # Invalidate DTM and labels state if rows were removed
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None
            if "count_vectorizer" in session.artifacts:
                del session.artifacts["count_vectorizer"]

        n_documents_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents_before": n_documents_before,
            "n_duplicates": n_duplicates,
            "dropped": drop,
            "n_dropped": n_dropped,
            "n_documents_after": n_documents_after
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary
        )

        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False
    ) -> Dict[str, Any]:
        """
        Subsamples rows from session.dataframe using pandas DataFrame.sample.
        
        Requires explicitly specifying either 'n' or 'frac'.
        Updates session.dataframe with the sampled subset and resets its index.
        Invalidates downstream DTM state (feature_matrix, feature_names, labels)
        to prevent dimensional mismatch.
        
        Args:
            n: Absolute number of rows to sample.
            frac: Fraction of rows to sample (0.0 to 1.0).
            random_state: Random seed for reproducible sampling.
            replace: Allow or disallow sampling of the same row more than once.
            
        Returns:
            Dict containing result_id, status, n_documents_before, n_documents_after,
            sampled_indices (preview of up to 10 original row indices sampled),
            total_sampled_indices, random_state, and a note.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        n_documents_before = len(session.dataframe)

        if n_documents_before == 0:
            return {
                "status": "error",
                "message": "session.dataframe is empty. Cannot sample from an empty dataset."
            }

        # Ensure user explicitly specifies either n or frac
        if n is None and frac is None:
            return {
                "status": "error",
                "message": "Please specify either 'n' (number of rows) or 'frac' (fraction of rows) to sample."
            }

        # Parameter validation
        if n is not None and n <= 0:
            return {
                "status": "error",
                "message": f"Sample size 'n' must be greater than 0, got {n}."
            }

        if frac is not None and frac <= 0.0:
            return {
                "status": "error",
                "message": f"Sample fraction 'frac' must be greater than 0.0, got {frac}."
            }

        if n is not None and n > n_documents_before and not replace:
            return {
                "status": "error",
                "message": f"Cannot sample n={n} rows from dataset with {n_documents_before} rows when replace=False."
            }

        sample_kwargs = {"replace": replace, "random_state": random_state}
        if n is not None:
            sample_kwargs["n"] = n
        elif frac is not None:
            sample_kwargs["frac"] = frac

        try:
            sampled_df = session.dataframe.sample(**sample_kwargs)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during sampling: {str(e)}"
            }

        # Capture original row indices BEFORE resetting index
        sampled_indices = sampled_df.index.tolist()

        # Update session.dataframe and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        # Invalidate downstream DTM and labels state
        session.feature_matrix = None
        session.feature_names = None
        session.labels = None
        if "count_vectorizer" in session.artifacts:
            del session.artifacts["count_vectorizer"]

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents_before": n_documents_before,
            "n_documents_after": len(session.dataframe),
            "sampled_indices": sampled_indices[:10],  # Return preview of up to 10 indices
            "total_sampled_indices": len(sampled_indices),
            "random_state": random_state,
            "note": "session.dataframe has been updated to the sampled subset. Downstream DTM state has been reset."
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={
                "n": n,
                "frac": frac,
                "random_state": random_state,
                "replace": replace
            },
            summary=summary
        )

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """
        Computes summary statistics on document text length and creates a box plot by category.
        
        Calculates text_length as character count (df["text"].apply(len)).
        Computes .describe() statistics overall and grouped by category_name.
        Generates a seaborn box plot set on session.pending_figure.
        
        Returns:
            Dict containing result_id, status, overall summary stats, and by_category summary stats.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "session.dataframe does not contain a 'text' column."
            }

        # Calculate character length per document text
        text_series = session.dataframe["text"].fillna("").astype(str)
        text_length = text_series.apply(len)

        # Compute overall describe statistics
        overall_stats = text_length.describe().to_dict()

        def _clean_stats_dict(d: dict) -> dict:
            cleaned = {}
            for k, v in d.items():
                if pd.isna(v) or np.isnan(v):
                    cleaned[k] = None
                else:
                    cleaned[k] = float(v) if isinstance(v, (np.floating, float)) else int(v)
            return cleaned

        overall_cleaned = _clean_stats_dict(overall_stats)

        # Compute describe statistics grouped by category_name (or category if category_name is missing)
        group_col = "category_name" if "category_name" in session.dataframe.columns else "category"
        by_category_cleaned = {}

        if group_col in session.dataframe.columns:
            df_temp = session.dataframe.copy()
            df_temp["text_length"] = text_length
            grouped = df_temp.groupby(group_col)["text_length"].describe()
            for cat_val, row in grouped.iterrows():
                by_category_cleaned[str(cat_val)] = _clean_stats_dict(row.to_dict())
        else:
            by_category_cleaned = {"all": overall_cleaned}

        # Create box plot matching visualize_result_tool figure pattern
        plot_df = session.dataframe.copy()
        plot_df["text_length"] = text_length

        fig, ax = plt.subplots(figsize=(8, 5))
        if group_col in plot_df.columns:
            sns.boxplot(data=plot_df, x=group_col, y="text_length", ax=ax)
            ax.set_xlabel(group_col)
        else:
            sns.boxplot(data=plot_df, y="text_length", ax=ax)

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution by Category")
        fig.tight_layout()

        # Close figure immediately after creation, matching visualize_result_tool pattern
        plt.close(fig)
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "overall": overall_cleaned,
            "by_category": by_category_cleaned
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary
        )

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool
    ]
