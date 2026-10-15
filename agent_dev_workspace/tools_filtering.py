from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Filters terms in the Document-Term Matrix (DTM) based on a minimum variance threshold.

        Terms with variance less than or equal to the threshold are flagged for removal.
        Does not mutate the global session.feature_matrix, ensuring other tools can still access
        all original features.

        Args:
            threshold (float): Minimum variance required to keep a feature. Default is 0.0.

        Returns:
            Dict[str, Any]: Summary containing total, kept, and removed counts, alongside sample terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix (DTM) not found in session. Please run build_dtm_tool first."
            }

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_documents, n_total_terms = X.shape

        if n_documents == 0 or n_total_terms == 0:
            return {
                "status": "error",
                "message": "Feature matrix is empty."
            }

        # Calculate population variance using Master's exact sum formula: (N * sum(X^2) - (sum(X))^2) / N^2
        # For integer DTM word counts, N * sum(X^2) - (sum(X))^2 is exact 0 for constant columns.
        N = n_documents
        if sparse.issparse(X):
            sum_X = np.array(X.sum(axis=0)).ravel()
            sum_X2 = np.array(X.power(2).sum(axis=0)).ravel()
        else:
            sum_X = np.sum(X, axis=0)
            sum_X2 = np.sum(X ** 2, axis=0)

        num_var = N * sum_X2 - (sum_X ** 2)
        num_var = np.maximum(num_var, 0.0)  # Clamp numerical floating noise at or below 0
        variances = num_var / (N ** 2)

        # Keep terms with variance > threshold (matching scikit-learn VarianceThreshold)
        keep_mask = variances > threshold
        kept_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]

        n_kept = int(len(kept_indices))
        n_removed = int(len(removed_indices))

        # Create full report DataFrame sorted by variance descending
        # Requirement: Column 1 is term/feature name, Column 2 is variance
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Store in session artifacts
        session.artifacts["variance_kept_terms"] = feature_names[kept_indices].tolist()
        session.artifacts["variance_report"] = df_report

        # Prepare concise summary (top kept terms and sample removed terms)
        sorted_kept_terms = df_report[df_report["term"].isin(feature_names[kept_indices])]["term"].tolist()
        sorted_removed_terms = df_report[df_report["term"].isin(feature_names[removed_indices])]["term"].tolist()

        top_kept_sample = sorted_kept_terms[:5]
        removed_sample = sorted_removed_terms[:5]

        result_id = session.next_result_id("variance")
        summary = {
            "result_id": result_id,
            "threshold": float(threshold),
            "n_total_terms": int(n_total_terms),
            "n_kept_terms": n_kept,
            "n_removed_terms": n_removed,
            "top_kept_terms_sample": top_kept_sample,
            "removed_terms_sample": removed_sample,
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculates Pearson correlation between term frequencies and a binary indicator
        for target_class (one-vs-rest). Stores full sorted results in session.
        
        Args:
            target_class (str): Target category name for one-vs-rest binary correlation.
            threshold (Optional[float]): Optional threshold on absolute Pearson r (|r| >= threshold)
                                         to filter selected features.
            top_n (Optional[int]): Optional limit to select top-N features by absolute correlation.
            
        Returns:
            Dict containing result_id, target_class, selected count, and top feature statistics.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {
                "status": "error",
                "message": "DTM or session.labels not initialized. Please load dataset and build DTM first."
            }
            
        # Verify target_class exists using session.categories / unique session.labels
        available_categories = list(session.categories) if session.categories else list(np.unique(session.labels))
            
        if target_class not in available_categories:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found in dataset.",
                "available_categories": available_categories
            }

        # 1. Construct binary indicator vector y aligned with session.labels
        labels = np.array(session.labels)
        y = (labels == target_class).astype(float)
        
        N = len(y)
        sum_y = np.sum(y)
        sum_y2 = np.sum(y ** 2)
        var_y_num = N * sum_y2 - (sum_y ** 2)
        
        X = session.feature_matrix  # scipy sparse matrix (N x M)
        feature_names = np.array(session.feature_names)
        
        if var_y_num <= 0:
            # Degenerate case: all target or no target
            r_values = np.zeros(len(feature_names))
        else:
            # 2. Exact sum formula for covariance and variance
            sum_X = np.array(X.sum(axis=0)).flatten()
            sum_X2 = np.array(X.power(2).sum(axis=0)).flatten()
            
            # Exact sum formulas
            num_var_X = np.maximum(N * sum_X2 - (sum_X ** 2), 0.0)
            std_X_num = np.sqrt(num_var_X)
            std_y_num = np.sqrt(var_y_num)
            
            # Cross term sum(X_j * y)
            sum_Xy = np.array(X.T.dot(y)).flatten()
            cov_num = N * sum_Xy - (sum_X * sum_y)
            
            denom = std_X_num * std_y_num
            
            # Handle zero variance / division by zero cleanly
            r_values = np.where(denom > 0, cov_num / denom, 0.0)
            r_values = np.nan_to_num(r_values, nan=0.0)

        # 3. Create full report DataFrame sorted by absolute correlation
        df_results = pd.DataFrame({
            "term": feature_names,
            "pearson_r": r_values,
            "abs_r": np.abs(r_values)
        })
        
        # Sort by absolute correlation descending
        df_results.sort_values(by="abs_r", ascending=False, inplace=True)
        
        # Prepare full_report according to spec: Column 1: term, Column 2: pearson_r
        full_report = df_results[["term", "pearson_r"]].copy()

        # 4. Determine selected features count based on threshold and top_n
        selected_df = df_results.copy()
        if threshold is not None:
            selected_df = selected_df[selected_df["abs_r"] >= threshold]
        if top_n is not None and top_n > 0:
            selected_df = selected_df.head(top_n)
            
        n_selected = len(selected_df)
        selected_sample = selected_df["term"].head(10).tolist()
        
        # Build summary dict
        result_id = session.next_result_id("pearson_filter")
        
        # Top 10 preview for LLM summary
        top_preview = df_results.head(10)
        top_features_dict = {
            row["term"]: float(np.round(row["pearson_r"], 4))
            for _, row in top_preview.iterrows()
        }
        
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_total_features": len(feature_names),
            "n_selected": n_selected,
            "top_selected_sample": selected_sample,
            "top_correlations": top_features_dict
        }
        
        session.store_result("pearson_filter_tool", {"target_class": target_class}, summary, full_report=full_report)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between term frequencies and a binary indicator
        for target_class (one-vs-rest). Stores full sorted results in session.

        Args:
            target_class (str): Target category name for one-vs-rest binary correlation.
            threshold (Optional[float]): Optional threshold on absolute Spearman r (|r| >= threshold)
                                         to filter selected features.
            top_n (Optional[int]): Optional limit to select top-N features by absolute correlation.

        Returns:
            Dict containing result_id, target_class, selected count, and top feature statistics.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {
                "status": "error",
                "message": "DTM or session.labels not initialized. Please load dataset and build DTM first."
            }

        available_categories = list(session.categories) if session.categories else list(np.unique(session.labels))

        if target_class not in available_categories:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found in dataset.",
                "available_categories": available_categories
            }

        labels = np.array(session.labels)
        y = (labels == target_class).astype(float)
        rank_y = rankdata(y, method="average")

        N = len(y)
        feature_names = np.array(session.feature_names)
        X = session.feature_matrix

        # Ensure CSC format for fast column slicing without dense conversion
        if sparse.issparse(X):
            X_csc = X.tocsc()
        else:
            X_csc = sparse.csc_matrix(X)

        n_terms = len(feature_names)

        # We construct a sparse rank matrix where zeros are shifted to 0.0 (offset = R_0)
        new_data_list = []
        new_indices_list = []
        new_indptr = [0]

        # Array to store global mean and variance for each column's full rank vector
        mean_R_X = np.zeros(n_terms, dtype=float)
        var_R_X = np.zeros(n_terms, dtype=float)

        for j in range(n_terms):
            start = X_csc.indptr[j]
            end = X_csc.indptr[j + 1]
            non_zero_vals = X_csc.data[start:end]
            non_zero_rows = X_csc.indices[start:end]
            nz_count = len(non_zero_vals)
            z_count = N - nz_count

            if nz_count == 0:
                # All zeros column
                # R_0 = (N + 1) / 2; shifted value = 0.0
                mean_R_X[j] = (N + 1) / 2.0
                var_R_X[j] = 0.0
                new_indptr.append(new_indptr[-1])
            else:
                # Rank non-zero elements locally
                local_ranks = rankdata(non_zero_vals, method="average")
                # Shifted values for non-zero elements: R_local + (z_count - 1) / 2.0
                shifted_nz_ranks = local_ranks + (z_count - 1.0) / 2.0

                new_data_list.append(shifted_nz_ranks)
                new_indices_list.append(non_zero_rows)
                new_indptr.append(new_indptr[-1] + nz_count)

                # Full vector mean is always (N + 1) / 2 for any rank vector without unassigned elements
                mean_R_X[j] = (N + 1) / 2.0

                # Compute exact population variance of the full rank vector:
                # sum((R - mean)^2) / N = (z_count * (R_0 - mean)^2 + sum((R_nz - mean)^2)) / N
                R_0 = (z_count + 1.0) / 2.0
                global_nz_ranks = local_ranks + z_count
                sum_sq_diff = z_count * ((R_0 - mean_R_X[j]) ** 2) + np.sum((global_nz_ranks - mean_R_X[j]) ** 2)
                var_R_X[j] = sum_sq_diff / N

        if len(new_data_list) > 0:
            all_data = np.concatenate(new_data_list)
            all_indices = np.concatenate(new_indices_list)
        else:
            all_data = np.array([], dtype=float)
            all_indices = np.array([], dtype=int)

        R_sparse = sparse.csc_matrix((all_data, all_indices, new_indptr), shape=(N, n_terms))

        # Pearson correlation on ranks (Spearman definition)
        # y_shift = rank_y - mean(rank_y)
        mean_y = (N + 1) / 2.0
        y_shift = rank_y - mean_y
        var_y = np.sum(y_shift ** 2) / N

        if var_y <= 0:
            r_values = np.zeros(n_terms)
        else:
            # Cross covariance: Cov(R_X, rank_y) = (R_sparse.T . y_shift) / N
            # Since R_sparse contains (R_X - R_0) and sum(y_shift) == 0,
            # R_sparse.T . y_shift == (R_X - R_0).T . y_shift == R_X.T . y_shift - R_0 * sum(y_shift) == R_X.T . y_shift
            cov_R_X_y = R_sparse.T.dot(y_shift) / N

            std_X = np.sqrt(np.maximum(var_R_X, 0.0))
            std_y = np.sqrt(var_y)
            denom = std_X * std_y

            r_values = np.where(denom > 0, cov_R_X_y / denom, 0.0)
            r_values = np.nan_to_num(r_values, nan=0.0)

        df_results = pd.DataFrame({
            "term": feature_names,
            "spearman_r": r_values,
            "abs_r": np.abs(r_values)
        })

        df_results.sort_values(by="abs_r", ascending=False, inplace=True)
        full_report = df_results[["term", "spearman_r"]].copy()

        selected_df = df_results.copy()
        if threshold is not None:
            selected_df = selected_df[selected_df["abs_r"] >= threshold]
        if top_n is not None and top_n > 0:
            selected_df = selected_df.head(top_n)

        n_selected = len(selected_df)
        selected_sample = selected_df["term"].head(10).tolist()

        result_id = session.next_result_id("spearman_filter")

        top_preview = df_results.head(10)
        top_features_dict = {
            row["term"]: float(np.round(row["spearman_r"], 4))
            for _, row in top_preview.iterrows()
        }

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_total_features": len(feature_names),
            "n_selected": n_selected,
            "top_selected_sample": selected_sample,
            "top_correlations": top_features_dict
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class}, summary, full_report=full_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
