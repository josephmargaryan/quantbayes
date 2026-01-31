import pandas as pd
from tqdm import tqdm


class SimilarityMatcher:
    def __init__(
        self,
        source_df,
        target_df,
        source_id,
        target_id,
        matching_config,
        overall_threshold=0.8,
    ):
        """
        source_df, target_df: DataFrames containing the records.
        source_id, target_id: Column names that uniquely identify records in each DataFrame.
        matching_config: Dictionary where each key is a column name (which must exist in both DataFrames)
                         and its value is a dictionary with:
                             - "sim_func": Function to compute similarity for the column.
                             - "weight": Weight (float) that contributes to the overall fuzzy similarity.
                             - "match_type": Either "fuzzy" or "direct".
                                            "fuzzy" means the similarity score is combined using its weight.
                                            "direct" means the function should return either 1 or 0.
                             - "override": Boolean flag. If True and this column (when using a direct match)
                                           returns 1.0, then the overall score is immediately set to 1.0.
        overall_threshold: The minimum overall similarity required to consider two records a match.
        """
        self.source_df = source_df
        self.target_df = target_df
        self.source_id = source_id
        self.target_id = target_id
        self.matching_config = matching_config
        self.overall_threshold = overall_threshold

    def calculate_similarity_for_column(self, a, b, sim_func):
        # Return 0.0 if either value is NaN or an empty string.
        if pd.isna(a) or pd.isna(b):
            return 0.0
        a_str = str(a).strip()
        b_str = str(b).strip()
        if not a_str or not b_str:
            return 0.0
        return sim_func(a, b)

    def match_records(self):
        matches = []
        total_comparisons = len(self.source_df) * len(self.target_df)
        pbar = tqdm(total=total_comparisons, desc="Matching records")
        for _, source_row in self.source_df.iterrows():
            for _, target_row in self.target_df.iterrows():
                weighted_sum = 0.0
                total_weight = 0.0
                override = False

                # For each configured column, calculate similarity
                for col, config in self.matching_config.items():
                    sim_func = config["sim_func"]
                    weight = config.get("weight", 0.0)
                    match_type = config.get("match_type", "fuzzy")
                    override_flag = config.get("override", False)

                    source_val = source_row[col]
                    target_val = target_row[col]

                    if match_type == "direct":
                        sim = SimilarityMetrics.exact_match(source_val, target_val)
                        # If override is True and we have an exact match, override overall similarity.
                        if override_flag and sim == 1.0:
                            override = True
                            break  # No need to check further columns.
                        # For non-override direct columns (like type), add their 1 or 0 multiplied by weight.
                        else:
                            weighted_sum += sim * weight
                            total_weight += weight
                    else:  # "fuzzy"
                        sim = self.calculate_similarity_for_column(
                            source_val, target_val, sim_func
                        )
                        weighted_sum += sim * weight
                        total_weight += weight

                overall_sim = (
                    1.0
                    if override
                    else (weighted_sum / total_weight if total_weight > 0 else 0.0)
                )

                if overall_sim >= self.overall_threshold:
                    matches.append(
                        {
                            "source_id": source_row[self.source_id],
                            "target_id": target_row[self.target_id],
                            "overall_similarity": overall_sim,
                        }
                    )
                pbar.update(1)
        pbar.close()
        return pd.DataFrame(matches)


###########################################################################
# Main Execution
###########################################################################
if __name__ == "__main__":
    # -----------------------------
    # Load in the preprocessed dataframes (from your preprocess.py output)
    # -----------------------------
    bayes = pd.read_parquet("preprocessed_bayes.parquet")
    veeva = pd.read_parquet("preprocessed_veeva.parquet")

    # -----------------------------
    # Define the matching configuration.
    #
    # In this configuration:
    # - "full_address", "name", and "phone" use fuzzy matching with weighted contributions.
    # - "type" uses a direct match via TypeMapper.type_similarity. It returns 1.0 if the veeva type
    #   maps correctly to the bayes type, and 0.0 otherwise. However, its result only contributes to
    #   the weighted sum and does not override the overall score.
    # - "email" uses an exact match and is set as an override. If an email match is found, the overall
    #   similarity is immediately set to 1.0.
    #
    # The keys "match_type" and "override" are strings and booleans that allow you to specify how each
    # column is treated. "fuzzy" means the similarity function’s result is weighted and combined,
    # "direct" means the function should yield either 1.0 or 0.0. The override flag (when True) causes
    # a direct match in that column to set the overall similarity to 1.0.
    # -----------------------------
    matching_config = {
        "full_address": {
            "sim_func": SimilarityMetrics.levenshtein_similarity,
            "weight": 0.4,
            "match_type": "fuzzy",
            "override": False,
        },
        "name": {
            "sim_func": SimilarityMetrics.levenshtein_similarity,
            "weight": 0.4,
            "match_type": "fuzzy",
            "override": False,
        },
        "type": {
            "sim_func": TypeMapper.type_similarity,
            "weight": 0.1,
            "match_type": "direct",
            "override": False,  # Its score (1 or 0) is combined by weight; it doesn't override overall.
        },
        "phone": {
            "sim_func": SimilarityMetrics.levenshtein_similarity,
            "weight": 0.1,
            "match_type": "fuzzy",
            "override": False,
        },
        "email": {
            "sim_func": SimilarityMetrics.exact_match,
            "weight": 0.0,  # Weight isn't used since it's an override field.
            "match_type": "direct",
            "override": True,  # A match here overrides the overall score to 1.0.
        },
    }

    # -----------------------------
    # Create the matcher and run the matching
    # -----------------------------
    matcher = SimilarityMatcher(
        source_df=bayes,
        target_df=veeva,
        source_id="id",  # Unique identifier column in bayes DataFrame.
        target_id="id",  # Unique identifier column in veeva DataFrame.
        matching_config=matching_config,
        overall_threshold=0.70,
    )

    match_results = matcher.match_records()
    print(match_results)

    # Optionally, save the results as a parquet file.
    match_results.to_parquet("results.parquet", index=False)
