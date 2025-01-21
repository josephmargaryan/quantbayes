import pandas as pd
from tqdm import tqdm

# import pyarrow.parquet as pq
# import pyarrow as pa


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
        similarity_methods,  # New: List of similarity functions/methods
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
        self.similarity_methods = similarity_methods
        self.threshold = threshold
        self.feature_threshold = feature_threshold
        self.output_file = output_file

    def calculate_similarity(self, a, b, method):
        """Calculate similarity using the specified method."""
        if pd.isna(a) or pd.isna(b):
            return 0
        return method(a, b)  # Use the method dynamically

    def match_records(self):
        matches = []
        with tqdm(
            total=len(self.source_df) * len(self.target_df), desc="Matching records"
        ) as pbar:
            for _, source_row in self.source_df.iterrows():
                for _, target_row in self.target_df.iterrows():
                    feature_similarities = []
                    total_similarity = 0
                    feature_threshold_met = False
                    for src_col, tgt_col, weight, method in zip(
                        self.source_columns,
                        self.target_columns,
                        self.feature_weights,
                        self.similarity_methods,
                    ):
                        similarity = self.calculate_similarity(
                            source_row[src_col], target_row[tgt_col], method
                        )
                        feature_similarities.append(similarity)
                        if similarity >= self.feature_threshold:
                            feature_threshold_met = True
                        total_similarity += similarity * weight

                    overall_similarity = total_similarity / sum(self.feature_weights)
                    if overall_similarity >= self.threshold and feature_threshold_met:
                        matches.append(
                            {
                                "source_id": source_row[self.source_id],
                                "target_id": target_row[self.target_id],
                                "overall_similarity": overall_similarity,
                                "feature_similarities": feature_similarities,  # Include breakdown
                            }
                        )
                    pbar.update(1)

        # Save results
        match_df = pd.DataFrame(matches)
        # pq.write_table(pa.Table.from_pandas(match_df), self.output_file)
        return match_df


if __name__ == "__main__":
    from similarity_tools.similarity_metrics import SimilarityMetrics

    # Example data
    source_df = pd.DataFrame({"id": [1, 2], "name": ["John Doe", "Jane Smith"]})
    target_df = pd.DataFrame({"id": [101, 102], "name": ["Jon Doe", "Jane Smyth"]})

    # SimilarityMatcher setup
    matcher = SimilarityMatcher(
        source_df=source_df,
        target_df=target_df,
        source_id="id",
        target_id="id",
        source_columns=["name"],
        target_columns=["name"],
        feature_weights=[1.0],
        similarity_methods=[SimilarityMetrics.cosine_similarity],
        threshold=0.8,
        feature_threshold=0.7,
    )

    matcher.match_records()
