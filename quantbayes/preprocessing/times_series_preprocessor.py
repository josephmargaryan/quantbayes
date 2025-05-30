import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class TimeSeriesPreprocessor:
    def __init__(
        self,
        datetime_col: str,
        target_col: str,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        feature_scaler: Optional[object] = None,  # e.g., StandardScaler()
        target_scaler: Optional[object] = None,  # e.g., MinMaxScaler()
        seq_length: int = 7,
        forecast_horizon: int = 1,
        top_k_fourier: int = 3,
        remove_na: bool = True,
        data_format: str = "dataframe",  # "dataframe" or "numpy"
        column_names: Optional[List[str]] = None,  # For numpy input
        target_index: Optional[
            int
        ] = None,  # For numpy input if target_col is not provided
        auto_detect: bool = True,  # Automatically detect categorical/numeric cols if not provided
    ):
        """
        Enhanced time-series preprocessor.

        Parameters
        ----------
        datetime_col : str
            Name of the datetime column.
        target_col : str
            Name of the target column.
        categorical_cols : Optional[List[str]]
            List of categorical column names.
        numeric_cols : Optional[List[str]]
            List of numeric column names.
        feature_scaler : object, optional
            A scikit-learn style scaler for features.
        target_scaler : object, optional
            A scikit-learn style scaler for the target.
        seq_length : int
            Number of timesteps per sequence.
        forecast_horizon : int
            How many steps ahead is the target.
        top_k_fourier : int
            Number of top Fourier components to add as features.
        remove_na : bool
            Whether to drop rows with NaNs.
        data_format : str
            "dataframe" or "numpy". For numpy input, column_names must be provided.
        column_names : Optional[List[str]]
            Column names for numpy array input.
        target_index : Optional[int]
            If using numpy input and target_col is not given, use target_index.
        auto_detect : bool
            If True, auto-detect categorical and numeric columns when not provided.
        """
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.categorical_cols = categorical_cols  # May be None; will auto-detect if so and auto_detect==True
        self.numeric_cols = (
            numeric_cols  # May be None; will auto-detect if so and auto_detect==True
        )
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.top_k_fourier = top_k_fourier
        self.remove_na = remove_na

        self.data_format = data_format.lower()
        self.column_names = column_names
        self.target_index = target_index
        self.auto_detect = auto_detect

        # Internal placeholders for fitted objects
        self.fitted_feature_scaler = None
        self.fitted_target_scaler = None
        self.fitted_encoders: Dict[str, OneHotEncoder] = {}

        # To store final feature columns (after one-hot encoding) – excluding datetime and target
        self.feature_columns_after_encoding: List[str] = []

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
        )

    def _prepare_dataframe(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert numpy input to DataFrame if needed."""
        if isinstance(data, np.ndarray):
            if self.column_names is None:
                raise ValueError("For numpy input, you must provide column_names.")
            data = pd.DataFrame(data, columns=self.column_names)
        return data.copy()

    def _auto_detect_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Auto-detect categorical and numeric columns (excluding target and datetime).
        Object or categorical dtypes are categorical; numeric columns with fewer than 10 unique values
        are considered categorical.
        """
        cat_cols, num_cols = [], []
        for col in df.columns:
            if col in [self.target_col, self.datetime_col]:
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

    def _remove_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with any NaN values if remove_na is True."""
        if self.remove_na:
            before = df.shape[0]
            df = df.dropna()
            after = df.shape[0]
            self.logger.info(f"Removed {before - after} rows with NaN values.")
        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical columns. If fit==True, fit new encoders.
        """
        for cat_col in self.categorical_cols:
            if cat_col not in df.columns:
                df[cat_col] = ""  # Fill missing categorical column
            if fit:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                reshaped = df[[cat_col]].astype(str).values
                encoded = encoder.fit_transform(reshaped)
                self.fitted_encoders[cat_col] = encoder
                self.logger.info(
                    f"Fitted one-hot encoder for '{cat_col}' with {len(encoder.categories_[0])} categories."
                )
            else:
                encoder = self.fitted_encoders[cat_col]
                reshaped = df[[cat_col]].astype(str).values
                encoded = encoder.transform(reshaped)
            new_cols = [f"{cat_col}__{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
            df = pd.concat([df.drop(columns=[cat_col]), encoded_df], axis=1)
        return df

    def _apply_scaling(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features and target.
        """
        if self.feature_scaler is not None:
            if fit:
                self.fitted_feature_scaler = self.feature_scaler.fit(X)
                X_scaled = self.fitted_feature_scaler.transform(X)
            else:
                X_scaled = self.fitted_feature_scaler.transform(X)
            self.logger.info("Applied feature scaling.")
        else:
            X_scaled = X

        if y is not None and self.target_scaler is not None:
            y = y.reshape(-1, 1)
            if fit:
                self.fitted_target_scaler = self.target_scaler.fit(y)
                y_scaled = self.fitted_target_scaler.transform(y).ravel()
            else:
                y_scaled = self.fitted_target_scaler.transform(y).ravel()
            self.logger.info("Applied target scaling.")
        else:
            y_scaled = y

        return X_scaled, y_scaled

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Adding temporal features from '{self.datetime_col}'.")
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
            self.logger.info(f"Converted '{self.datetime_col}' to datetime.")
        df["year"] = df[self.datetime_col].dt.year
        df["month"] = df[self.datetime_col].dt.month
        df["day_of_month"] = df[
            self.datetime_col
        ].dt.day  # Renamed from "day" to "day_of_month"
        df["day_of_week"] = df[self.datetime_col].dt.dayofweek
        df["day_of_year"] = df[self.datetime_col].dt.dayofyear
        df["quarter"] = df[self.datetime_col].dt.quarter
        if df[self.datetime_col].dt.hour.nunique() > 1:
            df["hour"] = df[self.datetime_col].dt.hour
        if df[self.datetime_col].dt.minute.nunique() > 1:
            df["minute"] = df[self.datetime_col].dt.minute
        if df[self.datetime_col].dt.second.nunique() > 1:
            df["second"] = df[self.datetime_col].dt.second
        self.logger.info("Temporal features added.")
        return df

    def _fourier_transform(
        self, series: np.ndarray, period: int = 365
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of a series and return frequencies and magnitudes.
        """
        n = len(series)
        fft_vals = np.fft.fft(series)
        freqs = np.fft.fftfreq(n)
        mags = np.abs(fft_vals)
        return freqs, mags

    def _add_fourier_features(
        self, df: pd.DataFrame, period: int = 365
    ) -> pd.DataFrame:
        """
        Add top_k_fourier Fourier features (global FFT on target series).
        """
        self.logger.info(
            f"Adding top {self.top_k_fourier} Fourier features using '{self.target_col}'."
        )
        target_series = df[self.target_col].values
        freqs, mags = self._fourier_transform(target_series, period=period)
        pos_idx = np.where(freqs > 0)
        freqs, mags = freqs[pos_idx], mags[pos_idx]
        top_idx = np.argsort(mags)[::-1][: self.top_k_fourier]
        for i, idx in enumerate(top_idx):
            feat_name = f"fft_feat_{i}"
            df[feat_name] = np.sin(
                2 * np.pi * freqs[idx] * np.arange(len(df))
            ) + np.cos(2 * np.pi * freqs[idx] * np.arange(len(df)))
            self.logger.debug(f"Added Fourier feature '{feat_name}'.")
        self.logger.info("Fourier features added.")
        return df

    def _create_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences of length seq_length with the subsequent forecast_horizon target.
        """
        seq_X, seq_y = [], []
        n_total = len(X)
        for i in range(n_total - self.seq_length - self.forecast_horizon + 1):
            seq_X.append(X[i : i + self.seq_length])
            if y is not None:
                seq_y.append(y[i + self.seq_length + self.forecast_horizon - 1])
        self.logger.info(f"Created {len(seq_X)} sequences.")
        X_seq = np.array(seq_X)
        y_seq = np.array(seq_y).reshape(-1, 1) if y is not None else None
        return X_seq, y_seq

    def fit_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        add_temporal: bool = True,
        add_fourier: bool = False,
        fourier_period: int = 365,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Fit the preprocessing pipeline on training data and return:
            - The processed DataFrame,
            - The sequences for X (shape: [num_sequences, seq_length, D]),
            - The sequences for y (shape: [num_sequences, 1]).
        """
        df = self._prepare_dataframe(data)
        df = df.sort_values(by=self.datetime_col).reset_index(drop=True)
        df = self._remove_nans(df)

        if self.auto_detect and (
            self.categorical_cols is None or self.numeric_cols is None
        ):
            auto_cat, auto_num = self._auto_detect_columns(df)
            if self.categorical_cols is None:
                self.categorical_cols = auto_cat
            if self.numeric_cols is None:
                self.numeric_cols = auto_num
            self.logger.info(
                f"Auto-detected categorical cols: {self.categorical_cols} and numeric cols: {self.numeric_cols}"
            )

        if add_temporal:
            df = self._add_temporal_features(df)
        if add_fourier:
            df = self._add_fourier_features(df, period=fourier_period)

        df = self._encode_categorical(df, fit=True)

        # Determine final feature columns (exclude datetime and target)
        all_cols = df.columns.tolist()
        self.feature_columns_after_encoding = [
            col for col in all_cols if col not in [self.target_col, self.datetime_col]
        ]

        X = df[self.feature_columns_after_encoding].values
        y = df[self.target_col].values

        X_scaled, y_scaled = self._apply_scaling(X, y, fit=True)

        df_processed = df.copy()
        df_processed[self.feature_columns_after_encoding] = X_scaled
        df_processed[self.target_col] = y_scaled

        X_seq, y_seq = self._create_sequences(
            df_processed[self.feature_columns_after_encoding].values,
            df_processed[self.target_col].values,
        )
        return df_processed, X_seq, y_seq

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        add_temporal: bool = True,
        add_fourier: bool = False,
        fourier_period: int = 365,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
        """
        Transform new (test/inference) data using the fitted preprocessing pipeline.
        If prediction_mode is True, the target column is not required.
        Returns:
            - The transformed DataFrame,
            - The sequences for X,
            - The sequences for y (or None if target is not available).
        """
        df = self._prepare_dataframe(data)
        df = df.sort_values(by=self.datetime_col).reset_index(drop=True)
        if self.remove_na:
            df = self._remove_nans(df)

        target_present = self.target_col in df.columns
        if not prediction_mode and not target_present:
            raise ValueError("Target column is missing and prediction_mode is False.")
        elif prediction_mode and not target_present:
            self.logger.info(
                "Prediction mode: target column not found; proceeding without target."
            )

        if add_temporal:
            df = self._add_temporal_features(df)
        if add_fourier and target_present:
            df = self._add_fourier_features(df, period=fourier_period)

        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = ""
        df = self._encode_categorical(df, fit=False)

        for col in self.feature_columns_after_encoding:
            if col not in df.columns:
                df[col] = 0

        cols_order = [self.datetime_col] + self.feature_columns_after_encoding
        if target_present:
            cols_order.append(self.target_col)
        df = df[cols_order]

        X = df[self.feature_columns_after_encoding].values
        y = df[self.target_col].values if target_present else None

        X_scaled, y_scaled = self._apply_scaling(X, y, fit=False)
        df[self.feature_columns_after_encoding] = X_scaled
        if target_present:
            df[self.target_col] = y_scaled

        if not target_present:
            y_scaled = None

        X_seq, y_seq = self._create_sequences(
            df[self.feature_columns_after_encoding].values,
            df[self.target_col].values if target_present else None,
        )
        return df, X_seq, y_seq

    def split_data(
        self, df: pd.DataFrame, train_size: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a processed DataFrame into train and test sets.
        """
        split_idx = int(len(df) * train_size)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def get_arrays(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from train and validation DataFrames and return arrays:
          X_train, X_val, y_train, y_val.
        """
        X_train, y_train = self._create_sequences(
            df_train[self.feature_columns_after_encoding].values,
            df_train[self.target_col].values,
        )
        X_val, y_val = self._create_sequences(
            df_val[self.feature_columns_after_encoding].values,
            df_val[self.target_col].values,
        )
        return X_train, X_val, y_train, y_val

    def plot_time_series(
        self,
        df: pd.DataFrame,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        freq: str = "D",
        title: str = "Time Series Plot",
    ):
        """
        Plot the time-series target between start_date and end_date, resampling by frequency.
        """
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        plot_df = df.copy()
        if start_date is not None:
            plot_df = plot_df[plot_df[self.datetime_col] >= pd.to_datetime(start_date)]
        if end_date is not None:
            plot_df = plot_df[plot_df[self.datetime_col] <= pd.to_datetime(end_date)]
        plot_df = plot_df.set_index(self.datetime_col)
        resampled = plot_df[self.target_col].resample(freq).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(resampled.index, resampled.values, marker="o", label=self.target_col)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.show()


# -------------------------------
# Extended Test Suite
# -------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 1) Time Series Processing (DataFrame input)
    date_rng = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
    n_samples = len(date_rng)
    feat1 = np.random.randn(n_samples) * 10 + 100
    feat2 = np.random.randn(n_samples) * 5 + 50
    cat_feat = np.random.choice(["A", "B", "C"], size=n_samples)
    t = np.arange(n_samples)
    target = (
        0.01 * t + 10 * np.sin(2 * np.pi * t / 365) + np.random.randn(n_samples) * 2
    )

    df_ts = pd.DataFrame(
        {
            "date": date_rng,
            "feat1": feat1,
            "feat2": feat2,
            "cat_feat": cat_feat,
            "target": target,
        }
    )

    print("=== Time Series Preprocessing (DataFrame) Demo ===")
    ts_processor = TimeSeriesPreprocessor(
        datetime_col="date",
        target_col="target",
        categorical_cols=["cat_feat"],
        numeric_cols=["feat1", "feat2"],
        feature_scaler=StandardScaler(),
        target_scaler=MinMaxScaler(),
        seq_length=7,
        forecast_horizon=1,
        top_k_fourier=3,
        remove_na=True,
        data_format="dataframe",
        auto_detect=False,
    )

    # Fit and transform training data; unpack processed DataFrame and sequences.
    proc_df, X_seq, y_seq = ts_processor.fit_transform(
        df_ts, add_temporal=True, add_fourier=True, fourier_period=365
    )
    print("Training sequences shapes:")
    print("X_seq:", X_seq.shape)  # Expected: [num_sequences, seq_length, D]
    print("y_seq:", y_seq.shape)  # Expected: [num_sequences, 1]

    # Test transform in prediction mode (without target column)
    df_ts_pred = df_ts.drop(columns=["target"])
    pred_df, X_seq_pred, y_seq_pred = ts_processor.transform(
        df_ts_pred, add_temporal=True, add_fourier=False, prediction_mode=True
    )
    print("Prediction mode sequences shape (X):", X_seq_pred.shape)
    print("Prediction mode y_seq (should be None):", y_seq_pred)

    # 2) Split Data and Get Arrays
    proc_df, _, _ = ts_processor.fit_transform(
        df_ts, add_temporal=True, add_fourier=True, fourier_period=365
    )
    train_df, test_df = ts_processor.split_data(proc_df, train_size=0.8)
    X_train, X_val, y_train, y_val = ts_processor.get_arrays(train_df, test_df)
    print("After splitting:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)

    # 3) Plotting Time Series
    print("Plotting the target time series (entire period):")
    ts_processor.plot_time_series(proc_df, freq="D", title="Time Series (Daily)")
    print("Plotting a zoomed-in view (June-August 2020):")
    ts_processor.plot_time_series(
        proc_df,
        start_date="2020-06-01",
        end_date="2020-08-01",
        freq="D",
        title="Zoomed-In Plot",
    )

    print("\nAll tests completed successfully.")


def test_preprocessor():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Create a small synthetic dataset
    date_rng = pd.date_range(start="2022-01-01", periods=15, freq="D")
    data = {
        "day": date_rng,  # datetime column
        "feature1": np.random.randn(15),
        "target": np.random.randint(0, 10, size=15),
    }
    df = pd.DataFrame(data)

    # Instantiate the preprocessor.
    # Note: auto_detect is True so it will automatically figure out that 'feature1' is numeric.
    processor = TimeSeriesPreprocessor(
        datetime_col="day",
        target_col="target",
        categorical_cols=None,
        numeric_cols=None,
        feature_scaler=StandardScaler(),
        target_scaler=None,
        seq_length=3,
        forecast_horizon=1,
        top_k_fourier=2,
        remove_na=True,
        data_format="dataframe",
        auto_detect=True,
    )

    try:
        # Test fit_transform on training data
        df_processed, X_seq, y_seq = processor.fit_transform(
            data=df,
            add_temporal=True,
            add_fourier=False,  # Disable Fourier to focus on temporal features
        )
        print("Fit Transform successful!")
        print("Processed DataFrame head:")
        print(df_processed.head())
        print("X_seq shape:", X_seq.shape)
        print("y_seq shape:", y_seq.shape)

        # Test transform in prediction mode (without target)
        df_new = df.drop(columns=["target"])
        test_df, test_X_seq, test_y_seq = processor.transform(
            data=df_new, add_temporal=True, add_fourier=False, prediction_mode=True
        )
        print("\nTransform (Prediction mode) successful!")
        print("Transformed Test DataFrame head:")
        print(test_df.head())
        print("Test X_seq shape:", test_X_seq.shape)
        print("Test y_seq (should be None):", test_y_seq)

    except Exception as e:
        print("Test failed with error:", e)


if __name__ == "__main__":
    test_preprocessor()
    # You can keep the other test code or demo below this block if needed.
