import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


class Preprocessor:
    """
    Enhanced Preprocessor for tabular data that supports regression, binary classification,
    and multiclass classification. It works with both pandas DataFrames and numpy arrays,
    can auto-detect categorical/numeric columns if not provided, and supports prediction mode
    when target values are missing.

    Parameters
    ----------
    task_type : str
        One of ["regression", "binary", "multiclass"].
    target_col : Optional[str]
        Name of the target column (if using DataFrame input). For numpy input, if None then target_index must be provided.
    categorical_cols : Optional[List[str]]
        List of categorical column names (for DataFrame input). If None and auto_detect is True, they will be auto-detected.
    numeric_cols : Optional[List[str]]
        List of numeric column names (for DataFrame input). If None and auto_detect is True, they will be auto-detected.
    feature_scaler : Optional[object]
        A scikit-learn style scaler for features.
    target_scaler : Optional[object]
        A scikit-learn style scaler for the target (applies only to regression).
    target_encoding : str, default="label"
        How to encode multiclass targets. Options:
        - "label": return 1-D integer labels.
        - "onehot": return a one-hot-encoded (n_samples × n_classes) array.
    remove_na : bool
        Whether to drop rows with any missing values.
    data_format : str
        Either "dataframe" or "numpy". Default is "dataframe".
    column_names : Optional[List[str]]
        For numpy input, a list of column names to convert the array into a DataFrame.
    target_index : Optional[int]
        For numpy input, the index corresponding to the target column (used if target_col is not provided).
    categorical_indices : Optional[List[int]]
        For numpy input, indices corresponding to categorical features.
    numeric_indices : Optional[List[int]]
        For numpy input, indices corresponding to numeric features.
    auto_detect : bool
        If True, auto-detect categorical and numeric columns (if not provided).
    """

    def __init__(
        self,
        task_type: str,
        target_col: Optional[str] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        feature_scaler=None,
        target_scaler=None,
        remove_na: bool = True,
        data_format: str = "dataframe",
        column_names: Optional[List[str]] = None,
        target_index: Optional[int] = None,
        categorical_indices: Optional[List[int]] = None,
        numeric_indices: Optional[List[int]] = None,
        auto_detect: bool = True,
        target_encoding: str = "label",
    ):
        self.task_type = task_type.lower()
        if self.task_type not in ["regression", "binary", "multiclass"]:
            raise ValueError(
                "task_type must be 'regression', 'binary', or 'multiclass'."
            )

        self.target_col = target_col
        self.categorical_cols = categorical_cols  # May be None
        self.numeric_cols = numeric_cols  # May be None
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.target_encoding = target_encoding.lower()
        if self.task_type == "multiclass" and self.target_encoding not in (
            "label",
            "onehot",
        ):
            raise ValueError(
                "For multiclass tasks, target_encoding must be 'label' or 'onehot'."
            )
        self.remove_na = remove_na

        # Additional parameters for numpy input support
        self.data_format = data_format.lower()
        self.column_names = column_names
        self.target_index = target_index
        self.categorical_indices = categorical_indices
        self.numeric_indices = numeric_indices
        self.auto_detect = auto_detect

        # Internal placeholders for fitted scalers/encoders
        self.fitted_feature_scaler = None
        self.fitted_target_scaler = None

        self.fitted_onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.fitted_label_encoder = None
        self.fitted_target_onehot_encoder = None

        # After encoding, track the feature column names
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

    def _encode_categorical_features(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """
        One-hot encode each categorical column using a separate OneHotEncoder.
        """
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = ""  # Create missing column with empty strings
            if fit:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_data = enc.fit_transform(df[[col]].astype(str))
                self.fitted_onehot_encoders[col] = enc
                self.logger.info(
                    f"Fitted one-hot encoder on column '{col}'. Categories: {enc.categories_}"
                )
            else:
                enc = self.fitted_onehot_encoders[col]
                encoded_data = enc.transform(df[[col]].astype(str))
            new_col_names = [f"{col}__{cat}" for cat in enc.categories_[0]]
            encoded_df = pd.DataFrame(
                encoded_data, columns=new_col_names, index=df.index
            )
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        return df

    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale numeric features if self.feature_scaler is provided."""
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
        - regression: optionally scales target.
        - binary: label-encode to 0/1.
        - multiclass: label-encode then one-hot encode.
        """
        if self.task_type == "regression":
            if self.target_scaler is not None:
                if fit:
                    self.fitted_target_scaler = self.target_scaler.fit(y.reshape(-1, 1))
                y_scaled = self.fitted_target_scaler.transform(y.reshape(-1, 1)).ravel()
                return y_scaled
            else:
                return y
        elif self.task_type == "binary":
            unique_vals = np.unique(y)
            if len(unique_vals) > 2:
                raise ValueError(
                    f"Binary task has more than 2 unique target values: {unique_vals}"
                )
            if fit:
                self.fitted_label_encoder = LabelEncoder().fit(y)
            y_encoded = self.fitted_label_encoder.transform(y)
            return y_encoded
        else:
            # multiclass: always label-encode; one-hot only if requested
            if fit:
                self.fitted_label_encoder = LabelEncoder().fit(y)
            y_int = self.fitted_label_encoder.transform(y)

            if self.target_encoding == "onehot":
                if fit:
                    self.fitted_target_onehot_encoder = OneHotEncoder(
                        sparse_output=False, handle_unknown="ignore"
                    ).fit(y_int.reshape(-1, 1))
                y_out = self.fitted_target_onehot_encoder.transform(
                    y_int.reshape(-1, 1)
                )
            else:
                # "label" encoding: return integer class labels
                y_out = y_int

            return y_out

    def _auto_detect_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect categorical and numeric columns.
        - Columns with object or category dtype are considered categorical.
        - Numeric columns with fewer than 10 unique values are also considered categorical.
        """
        cat_cols = []
        num_cols = []
        for col in df.columns:
            if col == self.target_col:
                continue
            if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 10:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
            else:
                cat_cols.append(col)
        return cat_cols, num_cols

    def _prepare_dataframe(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        If data is a numpy array and data_format is "numpy", convert it to a DataFrame using self.column_names.
        """
        if isinstance(data, np.ndarray):
            if self.column_names is None:
                raise ValueError("For numpy input, you must provide column_names.")
            data = pd.DataFrame(data, columns=self.column_names)
        return data.copy()

    def fit_transform(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit all preprocessing steps using the provided training data and transform it.
        If target values are missing, y is returned as None.
        Returns:
            X_processed (np.ndarray): Processed feature matrix.
            y_processed (np.ndarray or None): Processed target vector/array.
        """
        df = self._prepare_dataframe(data)
        df = self._remove_nans(df)

        # If using numpy input and target_col is not given, set it from target_index.
        if self.data_format == "numpy" and self.target_col is None:
            if self.target_index is None:
                raise ValueError(
                    "For numpy input, either target_col or target_index must be provided."
                )
            self.target_col = self.column_names[self.target_index]

        # If categorical/numeric columns are not provided and auto_detect is True, then auto-detect.
        if self.auto_detect:
            if self.categorical_cols is None or self.numeric_cols is None:
                auto_cat, auto_num = self._auto_detect_columns(df)
                if self.categorical_cols is None:
                    self.categorical_cols = auto_cat
                if self.numeric_cols is None:
                    self.numeric_cols = auto_num

        # For numpy input, if categorical_indices/numeric_indices are provided, override the above.
        if self.data_format == "numpy":
            if self.categorical_indices is not None:
                self.categorical_cols = [
                    self.column_names[i] for i in self.categorical_indices
                ]
            if self.numeric_indices is not None:
                self.numeric_cols = [self.column_names[i] for i in self.numeric_indices]

        # Ensure target_col exists
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data.")

        # One-hot encode categorical features (fit mode)
        df = self._encode_categorical_features(df, fit=True)

        # Store final feature column names (all columns except target)
        all_columns = df.columns.tolist()
        self.feature_columns_after_encoding = [
            c for c in all_columns if c != self.target_col
        ]

        # Separate features and target
        X = df[self.feature_columns_after_encoding].values
        y = df[self.target_col].values if self.target_col in df.columns else None

        X_scaled = self._scale_features(X, fit=True)
        y_processed = self._process_target(y, fit=True) if y is not None else None

        return X_scaled, y_processed

    def transform(
        self, data: Union[pd.DataFrame, np.ndarray], prediction_mode: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform new data using the fitted preprocessing steps.
        If prediction_mode is True, target values are not expected and y_processed will be None.
        Returns:
            X_processed (np.ndarray): Processed feature matrix.
            y_processed (np.ndarray or None): Processed target (if available and prediction_mode is False).
        """
        df = self._prepare_dataframe(data)
        df = self._remove_nans(df)

        # If not in prediction mode and target_col is missing, add a placeholder column.
        has_target = self.target_col in df.columns
        if not prediction_mode and not has_target:
            raise ValueError(
                "Target column is missing in the data and prediction_mode is False."
            )
        elif prediction_mode and not has_target:
            # For prediction, we simply continue without processing target.
            self.logger.info(
                "Prediction mode: target column not found; only transforming features."
            )

        # If categorical_cols were auto-detected during training, ensure they exist in new data.
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = ""

        df = self._encode_categorical_features(df, fit=False)

        # Make sure all feature columns exist (create missing ones as zeros)
        for col in self.feature_columns_after_encoding:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        target_list = [self.target_col] if self.target_col in df.columns else []
        df = df[self.feature_columns_after_encoding + target_list]

        X = df[self.feature_columns_after_encoding].values
        X_scaled = self._scale_features(X, fit=False)

        if not prediction_mode and self.target_col in df.columns:
            y = df[self.target_col].values
            y_processed = self._process_target(y, fit=False)
        else:
            y_processed = None

        return X_scaled, y_processed


#############################################
# Extended Test Cases for the Preprocessor
#############################################
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    # ---------------------------
    # 1) Regression scenario (DataFrame)
    # ---------------------------
    N = 100
    df_reg = pd.DataFrame(
        {
            "f1": np.random.randn(N),
            "f2": np.random.uniform(0, 100, size=N),
            "target": np.random.randn(N) * 50 + 10,
        }
    )
    print("=== Regression (DataFrame) Demo ===")
    preprocessor_reg = Preprocessor(
        task_type="regression",
        target_col="target",
        categorical_cols=[],  # no categorical features provided
        numeric_cols=["f1", "f2"],
        feature_scaler=StandardScaler(),
        target_scaler=MinMaxScaler(),
        remove_na=True,
        data_format="dataframe",
        auto_detect=False,  # already provided numeric cols
    )
    X_reg, y_reg = preprocessor_reg.fit_transform(df_reg)
    print("X_reg shape:", X_reg.shape, "y_reg shape:", y_reg.shape)

    # Test transform in prediction mode (without target)
    df_reg_pred = df_reg.drop(columns=["target"])
    X_reg_pred, y_reg_pred = preprocessor_reg.transform(
        df_reg_pred, prediction_mode=True
    )
    print(
        "Prediction mode (DataFrame) X_reg_pred shape:",
        X_reg_pred.shape,
        "y_reg_pred:",
        y_reg_pred,
    )

    # ---------------------------
    # 2) Binary classification (DataFrame) with auto-detection
    # ---------------------------
    df_bin = pd.DataFrame(
        {
            "cat_feature": np.random.choice(["A", "B", "C"], size=N),
            "num_feature": np.random.randn(N),
            "target": np.random.choice(["yes", "no"], size=N),
        }
    )
    print("\n=== Binary Classification (DataFrame) Demo ===")
    preprocessor_bin = Preprocessor(
        task_type="binary",
        target_col="target",
        categorical_cols=None,  # Let auto-detect handle this
        numeric_cols=None,
        feature_scaler=StandardScaler(),
        remove_na=True,
        data_format="dataframe",
        auto_detect=True,
    )
    X_bin, y_bin = preprocessor_bin.fit_transform(df_bin)
    print("X_bin shape:", X_bin.shape, "y_bin shape:", y_bin.shape)
    print("Unique target values after encoding:", np.unique(y_bin))

    # ---------------------------
    # 3) Multiclass classification (DataFrame)
    # ---------------------------
    df_multi = pd.DataFrame(
        {
            "feat_a": np.random.randn(N),
            "feat_b": np.random.uniform(0, 10, size=N),
            "cat_feat": np.random.choice(["X", "Y"], size=N),
            "target": np.random.choice(["cls1", "cls2", "cls3"], size=N),
        }
    )
    print("\n=== Multiclass Classification (DataFrame) Demo ===")
    preprocessor_multi = Preprocessor(
        task_type="multiclass",
        target_col="target",
        categorical_cols=None,
        numeric_cols=None,
        feature_scaler=MinMaxScaler(),
        remove_na=True,
        data_format="dataframe",
        auto_detect=True,
    )
    X_multi, y_multi = preprocessor_multi.fit_transform(df_multi)
    print("X_multi shape:", X_multi.shape, "y_multi shape:", y_multi.shape)
    print("First row of y_multi (one-hot):", y_multi[0])

    # ---------------------------
    # 4) Regression scenario (numpy array input)
    # ---------------------------
    # Create synthetic numpy data
    np_data = np.hstack(
        [
            np.random.randn(N, 2),  # two numeric features
            np.random.randn(N, 1),  # target column
        ]
    )
    # Define column names for conversion (f1, f2, target)
    col_names = ["f1", "f2", "target"]
    print("\n=== Regression (Numpy) Demo ===")
    preprocessor_np = Preprocessor(
        task_type="regression",
        target_col=None,  # not provided as string
        categorical_cols=None,  # will be auto-detected (likely none here)
        numeric_cols=None,  # auto-detect numeric features
        feature_scaler=StandardScaler(),
        target_scaler=MinMaxScaler(),
        remove_na=True,
        data_format="numpy",
        column_names=col_names,
        target_index=2,  # target is at index 2
        auto_detect=True,
    )
    X_np, y_np = preprocessor_np.fit_transform(np_data)
    print("X_np shape:", X_np.shape, "y_np shape:", y_np.shape)

    # Test transform in prediction mode for numpy input
    np_data_pred = np_data[:, :2]  # remove target column
    # For prediction, we still need to supply the same column names.
    preprocessor_np.column_names = [
        "f1",
        "f2",
    ]  # update column names for prediction mode
    X_np_pred, y_np_pred = preprocessor_np.transform(np_data_pred, prediction_mode=True)
    print(
        "Prediction mode (Numpy) X_np_pred shape:",
        X_np_pred.shape,
        "y_np_pred:",
        y_np_pred,
    )

    print("\nAll tests completed successfully.")
