"""
Pattern mining tools for frequent pattern discovery.
"""

import os
import tempfile
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth
from PAMI.extras.convert import DF2DB


def make_tools(session: Any) -> List[Any]:
    """
    Build pattern mining tools bound to the given session.

    Args:
        session: Shared SessionState instance.

    Returns:
        List of initialized LangChain tools.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mines frequent itemsets/patterns for a specific category using PAMI algorithms.

        Applies category-local text vectorization (with English stop words removed)
        and filters vocabulary based on variance, term_frequency, or tfidf before
        mining patterns using FPGrowth, Top-K (FAE), or MaxFPGrowth.

        Args:
            category_name: The category label to filter documents by (e.g. "catA").
            filtering_method: Vocabulary filter method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Absolute minimum support count (>= 1). Required for 'fpgrowth' and 'maxfpgrowth'.
            k: Number of highest-support patterns to keep (>= 1). Required for 'topk'.

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            num_patterns, and the mined patterns summary.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Call load_dataset_tool first."}

        # 1. Validate parameters
        filtering_method_clean = str(filtering_method).lower()
        if filtering_method_clean not in ["variance", "tfidf", "term_frequency"]:
            return {
                "status": "error",
                "message": f"Unsupported filtering_method: {filtering_method}. Must be 'variance', 'tfidf', or 'term_frequency'."
            }

        algorithm_clean = str(algorithm).lower()
        if algorithm_clean not in ["fpgrowth", "topk", "maxfpgrowth"]:
            return {
                "status": "error",
                "message": f"Unsupported algorithm: {algorithm}. Must be 'fpgrowth', 'topk', or 'maxfpgrowth'."
            }

        if algorithm_clean in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or not isinstance(min_sup, int) or min_sup < 1:
                return {
                    "status": "error",
                    "message": f"min_sup must be a positive integer >= 1 for algorithm '{algorithm_clean}'."
                }
        elif algorithm_clean == "topk":
            if k is None or not isinstance(k, int) or k < 1:
                return {
                    "status": "error",
                    "message": f"k must be a positive integer >= 1 for algorithm 'topk'."
                }

        df = session.dataframe
        if "category_name" not in df.columns:
            return {"status": "error", "message": "DataFrame missing 'category_name' column."}

        cat_df = df[df["category_name"] == category_name]
        if cat_df.empty:
            return {"status": "error", "message": f"Category '{category_name}' not found in dataset or has no documents."}

        result_id = session.next_result_id("mine_patterns")
        artifact_key = f"patterns_{category_name}_{filtering_method_clean}_{algorithm_clean}"

        tool_args = {
            "category_name": category_name,
            "filtering_method": filtering_method_clean,
            "algorithm": algorithm_clean,
            "min_sup": min_sup,
            "k": k
        }

        # 2. Build Category-Local DTM (CountVectorizer with stop_words='english')
        texts = cat_df["text"].fillna("")
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                summary = {
                    "result_id": result_id,
                    "category_name": category_name,
                    "filtering_method": filtering_method_clean,
                    "algorithm": algorithm_clean,
                    "num_patterns": 0,
                    "patterns": [],
                    "note": "empty vocabulary after stop word removal"
                }
                session.artifacts[artifact_key] = []
                session.store_result("mine_patterns_tool", tool_args, summary)
                return summary
            return {"status": "error", "message": f"Vectorization error: {str(e)}"}

        feature_names = np.array(vectorizer.get_feature_names_out())
        dtm_dense = dtm.toarray()

        if dtm_dense.shape[1] == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method_clean,
                "algorithm": algorithm_clean,
                "num_patterns": 0,
                "patterns": [],
                "note": "no terms in vocabulary"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", tool_args, summary)
            return summary

        # 3. Apply Filtering Method
        if filtering_method_clean == "variance":
            scores = np.var(dtm_dense, axis=0)
            low_thresh = np.quantile(scores, 0.05)
            high_thresh = np.quantile(scores, 0.95)
            mask = (scores >= low_thresh) & (scores <= high_thresh)
        elif filtering_method_clean == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)
            low_thresh = np.quantile(scores, 0.05)
            high_thresh = np.quantile(scores, 0.95)
            mask = (scores >= low_thresh) & (scores <= high_thresh)
        elif filtering_method_clean == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm_dense).toarray()
            scores = np.mean(tfidf_matrix, axis=0)
            low_thresh = np.quantile(scores, 0.20)
            mask = scores >= low_thresh

        kept_indices = np.where(mask)[0]
        kept_terms = list(feature_names[kept_indices])

        # Step 3 check: Handling 0 surviving terms cleanly
        if len(kept_terms) == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method_clean,
                "algorithm": algorithm_clean,
                "num_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", tool_args, summary)
            return summary

        filtered_dtm = dtm_dense[:, kept_indices]
        tx_df = pd.DataFrame(filtered_dtm, columns=kept_terms)

        # 4. Convert to Transactional DB file using DF2DB and tempfile
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        temp_db_path = temp_file.name
        temp_file.close()

        try:
            try:
                converter = DF2DB.DF2DB(tx_df)
                converter.convert2TransactionalDatabase(temp_db_path, ">=", 1)
            except Exception as e:
                return {"status": "error", "message": f"DF2DB conversion error: {str(e)}"}

            # 5. Run PAMI Mining Algorithm
            try:
                if algorithm_clean == "fpgrowth":
                    model = FPGrowth.FPGrowth(iFile=temp_db_path, minSup=min_sup, sep="\t")
                elif algorithm_clean == "topk":
                    model = FAE.FAE(iFile=temp_db_path, k=k, sep="\t")
                elif algorithm_clean == "maxfpgrowth":
                    model = MaxFPGrowth.MaxFPGrowth(iFile=temp_db_path, minSup=min_sup, sep="\t")

                model.mine()
                patterns_df = model.getPatternsAsDataFrame()
            except Exception as e:
                return {"status": "error", "message": f"PAMI mining error: {str(e)}"}

            # 6. Parse and format patterns
            patterns_list = []
            if patterns_df is not None and not patterns_df.empty:
                pat_col = patterns_df.columns[0]
                sup_col = patterns_df.columns[1]

                for _, row in patterns_df.iterrows():
                    raw_pat = str(row[pat_col]).strip()
                    pat_items = [item for item in raw_pat.split() if item]
                    sup_val = int(row[sup_col])
                    patterns_list.append({
                        "pattern": pat_items,
                        "support": sup_val
                    })

            # Sort patterns descending by support
            patterns_list.sort(key=lambda x: x["support"], reverse=True)

            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method_clean,
                "algorithm": algorithm_clean,
                "num_patterns": len(patterns_list),
                "patterns": patterns_list[:20]  # Top-20 for summary to keep JSON small
            }

            session.artifacts[artifact_key] = patterns_list
            full_report_df = pd.DataFrame(patterns_list) if patterns_list else pd.DataFrame(columns=["pattern", "support"])

            session.store_result("mine_patterns_tool", tool_args, summary, full_report=full_report_df)
            return summary

        finally:
            if os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except OSError:
                    pass

    return [mine_patterns_tool]
