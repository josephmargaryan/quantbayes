import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa


class SimilarityMatcher:
    def __init__(
        self,
        source_df,
        target_df,
        source_id,
        target_id,
        source_columns,
        target_columns,
        feature_weights,
        threshold,
        feature_threshold,
        output_file="results.parquet",
    ):
        self.source_df = source_df
        self.target_df = target_df
        self.source_id = source_id
        self.target_id = target_id
        self.source_columns = source_columns
        self.target_columns = target_columns
        self.feature_weights = feature_weights
        self.threshold = threshold
        self.feature_threshold = feature_threshold
        self.output_file = output_file

    def calculate_similarity(self, a, b):
        """Calculate similarity between two values using Levenshtein ratio."""
        if pd.isna(a) or pd.isna(b):
            return None
        return fuzz.ratio(str(a), str(b)) / 100

    def match_records(self):
        matches = []
        with tqdm(
            total=len(self.source_df) * len(self.target_df), desc="Matching records"
        ) as pbar:
            for _, source_row in self.source_df.iterrows():
                for _, target_row in self.target_df.iterrows():
                    total_similarity = 0
                    feature_threshold_met = False
                    for src_col, tgt_col, weight in zip(
                        self.source_columns, self.target_columns, self.feature_weights
                    ):
                        similarity = self.calculate_similarity(
                            source_row[src_col], target_row[tgt_col]
                        )
                        if similarity and similarity >= self.feature_threshold:
                            feature_threshold_met = True
                        if similarity:
                            total_similarity += similarity * weight

                    # Normalize similarity and check thresholds
                    overall_similarity = total_similarity / sum(self.feature_weights)
                    if overall_similarity >= self.threshold and feature_threshold_met:
                        matches.append(
                            {
                                "source_id": source_row[self.source_id],
                                "target_id": target_row[self.target_id],
                                "overall_similarity": overall_similarity,
                            }
                        )
                    pbar.update(1)

        # Save results to parquet file
        match_df = pd.DataFrame(matches)
        pq.write_table(pa.Table.from_pandas(match_df), self.output_file)
