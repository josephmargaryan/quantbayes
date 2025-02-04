import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class Preprocessor:
    """
    A single Preprocessor class that can handle regression, binary classification,
    and multiclass classification, based on 'task_type'.

    For classification, optionally encodes the target:
      - binary -> shape (N,)
      - multiclass -> shape (N, num_classes) if one-hot

    For regression, keeps target as numeric shape (N,).

    Also handles:
      - Missing data removal (optional)
      - Categorical encoding (one-hot)
      - Numeric scaling
      - Target scaling (for regression) or encoding (for classification)

    Parameters
    ----------
    task_type : str
        One of ["regression", "binary", "multiclass"].
    target_col : str
        Name of the target column in the DataFrame.
    categorical_cols : Optional[List[str]]
        List of categorical column names.
    numeric_cols : Optional[List[str]]
        List of numeric column names.
    feature_scaler : Optional[object]
        Any scikit-learn style scaler (e.g., StandardScaler) for features.
    target_scaler : Optional[object]
        Any scikit-learn style scaler for target (only applies to regression).
    remove_na : bool
        Whether to drop rows with any NaN values.
    """

    def __init__(
        self,
        task_type: str,
        target_col: str,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        feature_scaler=None,
        target_scaler=None,
        remove_na: bool = True,
    ):
        self.task_type = task_type.lower()
        if self.task_type not in ["regression", "binary", "multiclass"]:
            raise ValueError("task_type must be 'regression', 'binary', or 'multiclass'.")

        self.target_col = target_col
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.remove_na = remove_na

        # Internal placeholders
        self.fitted_feature_scaler = None
        self.fitted_target_scaler = None  # Used only if regression & target_scaler != None

        # For categorical encoding
        #   We'll do one-hot encoding for features
        self.fitted_onehot_encoders: Dict[str, OneHotEncoder] = {}

        # For classification target encoding
        #   - For binary, if not {0,1}, we label-encode to 0,1
        #   - For multiclass, we label-encode then one-hot
        self.fitted_label_encoder = None
        self.fitted_target_onehot_encoder = None

        # We will store final column order after one-hot encoding
        self.feature_columns_after_encoding: List[str] = []

        # Logger setup
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
        )

    def _remove_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with any NaN values if self.remove_na is True."""
        if self.remove_na:
            na_count_before = df.isna().sum().sum()
            df = df.dropna()
            na_count_after = df.isna().sum().sum()
            self.logger.info(
                f"Removed {na_count_before - na_count_after} NaN cells from the DataFrame."
            )
        return df

    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode each categorical column with a separate OneHotEncoder.
        """
        for col in self.categorical_cols:
            if col not in df.columns:
                # If the column is missing, create it with an empty string (or some placeholder).
                df[col] = ""

            if fit:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_data = enc.fit_transform(df[[col]].astype(str))
                self.fitted_onehot_encoders[col] = enc
                self.logger.info(f"Fitted one-hot encoder on column '{col}'. "
                                 f"Found categories: {enc.categories_}")
            else:
                enc = self.fitted_onehot_encoders[col]
                encoded_data = enc.transform(df[[col]].astype(str))

            # Construct the new column names
            new_col_names = [
                f"{col}__{cat}" for cat in self.fitted_onehot_encoders[col].categories_[0]
            ]
            # Convert to DataFrame
            encoded_df = pd.DataFrame(encoded_data, columns=new_col_names, index=df.index)
            # Drop original column
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        return df

    def _scale_features(self, X, fit: bool = False):
        """
        Scale numeric features if self.feature_scaler is provided.
        """
        if self.feature_scaler is not None:
            if fit:
                self.fitted_feature_scaler = self.feature_scaler.fit(X)
                return self.fitted_feature_scaler.transform(X)
            else:
                return self.fitted_feature_scaler.transform(X)
        return X

    def _process_target(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Process (encode/scale) the target based on the task_type.
        - regression: optional scaling
        - binary: label-encode if needed (0/1), final shape (N,)
        - multiclass: label-encode -> one-hot, final shape (N, num_classes)
        """
        if self.task_type == "regression":
            # Optionally scale target
            if self.target_scaler is not None:
                if fit:
                    self.fitted_target_scaler = self.target_scaler.fit(y.reshape(-1, 1))
                y_scaled = self.fitted_target_scaler.transform(y.reshape(-1, 1)).ravel()
                return y_scaled
            else:
                return y

        elif self.task_type == "binary":
            # We want y in shape (N,), with 0/1
            # Let's ensure it is numeric by label-encoding if needed
            # but only if distinct values > 2 we raise error
            unique_vals = np.unique(y)
            if len(unique_vals) > 2:
                raise ValueError("Binary task has more than 2 unique target values. "
                                 f"Found: {unique_vals}")
            # If the target is something like {'yes', 'no'} or {2, 5}, we map to {0, 1}
            if fit:
                self.fitted_label_encoder = LabelEncoder().fit(y)
            y_encoded = self.fitted_label_encoder.transform(y)
            return y_encoded

        else:
            # Multiclass => label-encode -> one-hot => final shape (N, num_classes)
            if fit:
                self.fitted_label_encoder = LabelEncoder().fit(y)
            y_int = self.fitted_label_encoder.transform(y)

            # One-hot the integer classes
            if fit:
                self.fitted_target_onehot_encoder = OneHotEncoder(sparse_output=False).fit(
                    y_int.reshape(-1, 1)
                )
            y_ohe = self.fitted_target_onehot_encoder.transform(y_int.reshape(-1, 1))
            return y_ohe

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit all preprocessing steps using this df as "training data",
        then transform the df and return (X, y).

        X shape:
          - after numeric scaling and one-hot encoding of categorical features => (N, D')

        y shape:
          - if regression or binary => (N,)
          - if multiclass => (N, num_classes)

        Returns
        -------
        X_processed, y_processed
        """
        df = df.copy()

        # Remove any missing data
        df = self._remove_nans(df)

        # One-hot encode categorical features
        df = self._encode_categorical_features(df, fit=True)

        # Keep track of which columns are features vs. target
        #   after encoding, numeric_cols might have changed. We'll retrieve them from df minus target col
        all_columns = df.columns.tolist()
        if self.target_col not in all_columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")
        self.feature_columns_after_encoding = [c for c in all_columns if c != self.target_col]

        # Separate X, y
        X = df[self.feature_columns_after_encoding].values
        y = df[self.target_col].values

        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        # Process target
        y_processed = self._process_target(y, fit=True)

        return X_scaled, y_processed

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """
        Transform new data (i.e., test / validation) using the previously fitted transformations.

        Returns X, y where:
          - y might be None if target_col doesn't exist in df
          - otherwise y is processed similarly to fit_transform
        """
        df = df.copy()

        # If target is missing, we add a placeholder column
        has_target = self.target_col in df.columns
        if not has_target:
            df[self.target_col] = np.nan  # dummy

        # Remove NA if desired
        df = self._remove_nans(df)

        # One-hot encode categorical features
        #   (we do NOT fit again)
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = ""
        df = self._encode_categorical_features(df, fit=False)

        # Make sure all feature columns exist
        for col in self.feature_columns_after_encoding:
            if col not in df.columns:
                # if missing, create zero column
                df[col] = 0

        # Reorder columns so that feature columns are consistent
        #   (some columns may appear in a different order after concatenation)
        df = df[self.feature_columns_after_encoding + [self.target_col]]

        # Separate X, y
        X = df[self.feature_columns_after_encoding].values
        y = df[self.target_col].values

        # Scale features
        X_scaled = self._scale_features(X, fit=False)

        # Process target (only if it exists)
        if has_target:
            y_processed = self._process_target(y, fit=False)
        else:
            y_processed = None

        return X_scaled, y_processed


if __name__ == "__main__":
    """
    Test section:
      Demonstrates usage for:
       1) Regression
       2) Binary classification
       3) Multiclass classification
    We'll generate synthetic data for each scenario and show how to use Preprocessor.
    """

    import numpy as np
    import pandas as pd

    # For reproducibility
    np.random.seed(42)

    # -------------------------------------------------
    # 1) Regression scenario
    # -------------------------------------------------
    N = 100
    df_reg = pd.DataFrame({
        "feature1": np.random.randn(N),
        "feature2": np.random.uniform(0, 100, size=N),
        "target": np.random.randn(N) * 50 + 10  # some numeric target
    })

    print("=== Regression Demo ===")
    preprocessor_reg = Preprocessor(
        task_type="regression",
        target_col="target",
        categorical_cols=[],  # none in this dataset
        numeric_cols=["feature1", "feature2"],
        feature_scaler=StandardScaler(),
        target_scaler=MinMaxScaler(),
        remove_na=True
    )
    X_reg, y_reg = preprocessor_reg.fit_transform(df_reg)
    print("Shapes:", X_reg.shape, y_reg.shape)
    print("X_reg (first 5 rows):\n", X_reg[:5])
    print("y_reg (first 5):\n", y_reg[:5], "\n")

    # Simulate "test" or "new" data
    df_reg_test = pd.DataFrame({
        "feature1": np.random.randn(5),
        "feature2": np.random.uniform(0, 100, size=5),
        "target": np.random.randn(5) * 50 + 10  # some numeric target
    })
    X_reg_test, y_reg_test = preprocessor_reg.transform(df_reg_test)
    print("X_reg_test shape:", X_reg_test.shape, "y_reg_test shape:", y_reg_test.shape)
    print("X_reg_test:\n", X_reg_test)
    print("y_reg_test:\n", y_reg_test, "\n")

    # -------------------------------------------------
    # 2) Binary classification scenario
    # -------------------------------------------------
    N = 100
    df_bin = pd.DataFrame({
        "feature_cat": np.random.choice(["A", "B", "C"], size=N),
        "feature_num": np.random.randn(N),
        "target": np.random.choice(["yes", "no"], size=N)  # or could be {0, 1}
    })

    print("=== Binary Classification Demo ===")
    preprocessor_bin = Preprocessor(
        task_type="binary",
        target_col="target",
        categorical_cols=["feature_cat"],
        numeric_cols=["feature_num"],
        feature_scaler=StandardScaler(),
        target_scaler=None,  # not used for binary
        remove_na=True
    )
    X_bin, y_bin = preprocessor_bin.fit_transform(df_bin)
    print("Shapes:", X_bin.shape, y_bin.shape)
    print("X_bin (first 5 rows):\n", X_bin[:5])
    print("y_bin (first 5):\n", y_bin[:5], "\n")
    print("Unique target values after encoding:", np.unique(y_bin))

    # Simulate "new" data
    df_bin_test = pd.DataFrame({
        "feature_cat": np.random.choice(["A", "B", "C"], size=5),
        "feature_num": np.random.randn(5),
        "target": np.random.choice(["yes", "no"], size=5)
    })
    X_bin_test, y_bin_test = preprocessor_bin.transform(df_bin_test)
    print("X_bin_test shape:", X_bin_test.shape, "y_bin_test shape:", y_bin_test.shape)
    print("X_bin_test:\n", X_bin_test)
    print("y_bin_test:\n", y_bin_test, "\n")

    # -------------------------------------------------
    # 3) Multiclass classification scenario
    # -------------------------------------------------
    N = 100
    df_multi = pd.DataFrame({
        "feat_a": np.random.randn(N),
        "feat_b": np.random.uniform(0, 10, size=N),
        "category": np.random.choice(["X", "Y"], size=N),
        "target": np.random.choice(["cls1", "cls2", "cls3"], size=N)
    })

    print("=== Multiclass Classification Demo ===")
    preprocessor_multi = Preprocessor(
        task_type="multiclass",
        target_col="target",
        categorical_cols=["category"],
        numeric_cols=["feat_a", "feat_b"],
        feature_scaler=MinMaxScaler(),
        target_scaler=None,
        remove_na=True
    )
    X_multi, y_multi = preprocessor_multi.fit_transform(df_multi)
    print("Shapes:", X_multi.shape, y_multi.shape)
    print("X_multi (first 5 rows):\n", X_multi[:5])
    print("y_multi (first 5):\n", y_multi[:5], "\n")
    print("Shape of y_multi:", y_multi.shape, "(should be (N, num_classes)).")
    print("Sum of first row of y_multi (should be 1):", y_multi[0].sum())

    # "New" data
    df_multi_test = pd.DataFrame({
        "feat_a": np.random.randn(5),
        "feat_b": np.random.uniform(0, 10, size=5),
        "category": np.random.choice(["X", "Y"], size=5),
        "target": np.random.choice(["cls1", "cls2", "cls3"], size=5)
    })
    X_multi_test, y_multi_test = preprocessor_multi.transform(df_multi_test)
    print("X_multi_test shape:", X_multi_test.shape, "y_multi_test shape:", y_multi_test.shape)
    print("X_multi_test:\n", X_multi_test)
    print("y_multi_test:\n", y_multi_test)

    print("\nAll tests completed successfully.")
