import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch

from typing import Optional, List, Tuple, Union
from datetime import datetime
import logging

class TimeSeriesPreprocessor:
    def __init__(
        self,
        datetime_col: str,
        target_col: str,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        feature_scaler=None,  # e.g., StandardScaler() for features
        target_scaler=None,  # e.g., MinMaxScaler() for target
        seq_length: int = 7,  # how many time steps in each input sequence
        forecast_horizon: int = 1,  # how many steps ahead to forecast
        top_k_fourier: int = 3,  # number of top Fourier components
        remove_na: bool = True,
    ):
        """
        A comprehensive time-series preprocessing class.

        :param datetime_col: str - the name of the datetime column in the DataFrame
        :param target_col: str - the name of the target column in the DataFrame
        :param categorical_cols: list of str - columns that should be treated as categorical
        :param numeric_cols: list of str - columns that should be treated as numeric
        :param feature_scaler: scaler object for features (e.g., StandardScaler(), None if no scaling)
        :param target_scaler: scaler object for target (e.g., MinMaxScaler(), None if no scaling)
        :param seq_length: int - number of timesteps to include in each input sequence
        :param forecast_horizon: int - how many steps ahead is the target
        :param top_k_fourier: int - how many top Fourier components to add as features
        :param remove_na: bool - whether to drop rows with NaN
        """
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.top_k_fourier = top_k_fourier
        self.remove_na = remove_na

        # Internal placeholders
        self.fitted_feature_scaler = None
        self.fitted_target_scaler = None
        self.fitted_encoders = {}
        self.feature_columns_after_encoding = None

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
        )

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encodes categorical columns. If fit=True, fit the OneHotEncoders.
        Otherwise, uses existing fitted encoders.
        """
        for cat_col in self.categorical_cols:
            original_unique_values = df[cat_col].nunique()
            if fit:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                reshaped = df[[cat_col]].astype(str).values
                encoded_data = encoder.fit_transform(reshaped)
                self.fitted_encoders[cat_col] = encoder
                self.logger.info(
                    f"Fitted encoder for column '{cat_col}' with {original_unique_values} unique values."
                )
            else:
                encoder = self.fitted_encoders[cat_col]
                reshaped = df[[cat_col]].astype(str).values
                encoded_data = encoder.transform(reshaped)

            self.logger.debug(
                f"One-hot encoding applied to column '{cat_col}', resulting in {encoded_data.shape[1]} new columns."
            )
            # Create column names for the encoded data
            encoder_col_names = [
                f"{cat_col}__{cat_class}"
                for cat_class in self.fitted_encoders[cat_col].categories_[0]
            ]

            # Create a DataFrame with encoded columns
            encoded_df = pd.DataFrame(
                encoded_data, columns=encoder_col_names, index=df.index
            )
            # Drop the original column and concatenate the new one-hot columns
            df = pd.concat([df.drop(columns=[cat_col]), encoded_df], axis=1)

        return df

    def _remove_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with any NaN values if self.remove_na is True"""
        if self.remove_na:
            df = df.dropna()
            na_count_before = df.isna().sum().sum()
            na_columns = df.columns[df.isna().any()].tolist()
            df = df.dropna()
            na_count_after = df.isna().sum().sum()
            self.logger.info(
                f"Removed {na_count_before - na_count_after} NaNs from columns: {na_columns}"
            )
        return df

    def _apply_scaling(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, fit: bool = False
    ):
        """
        Apply scaling to features and target.
        If fit=True, fit the scalers first, then transform.
        """
        # Scale features
        self.logger.debug(f"Applying scaling to features of shape {X.shape}.")
        if y is not None:
            self.logger.debug(f"Target scaling applied to target of shape {y.shape}.")

        if self.feature_scaler is not None:
            if fit:
                self.fitted_feature_scaler = self.feature_scaler.fit(X)
                X_scaled = self.fitted_feature_scaler.transform(X)
                self.logger.info("Feature scaler fitted and applied.")
            else:
                X_scaled = self.fitted_feature_scaler.transform(X)
        else:
            X_scaled = X

        # Scale target
        if y is not None and self.target_scaler is not None:
            y = y.reshape(-1, 1)  # reshape for scaler
            if fit:
                self.fitted_target_scaler = self.target_scaler.fit(y)
                y_scaled = self.fitted_target_scaler.transform(y).ravel()
                self.logger.info("Target scaler fitted and applied.")
            else:
                y_scaled = self.fitted_target_scaler.transform(y).ravel()
        else:
            y_scaled = y

        return X_scaled, y_scaled

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically add temporal features based on the granularity of the datetime column.
        Determines appropriate features dynamically based on available datetime attributes.

        :param df: DataFrame with a datetime column.
        :return: DataFrame with added temporal features.
        """
        self.logger.info(f"Adding temporal features from column '{self.datetime_col}'.")

        # Ensure datetime_col is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
            self.logger.info(f"Converted '{self.datetime_col}' to datetime type.")

        # Add year
        if hasattr(df[self.datetime_col].dt, "year"):
            df["year"] = df[self.datetime_col].dt.year
            self.logger.debug("Added 'year' feature.")

        # Add month
        if hasattr(df[self.datetime_col].dt, "month"):
            df["month"] = df[self.datetime_col].dt.month
            self.logger.debug("Added 'month' feature.")

        # Add day
        if hasattr(df[self.datetime_col].dt, "day"):
            df["day"] = df[self.datetime_col].dt.day
            self.logger.debug("Added 'day' feature.")

        # Add day of week
        if hasattr(df[self.datetime_col].dt, "dayofweek"):
            df["day_of_week"] = df[self.datetime_col].dt.dayofweek
            self.logger.debug("Added 'day_of_week' feature.")

        # Add day of year
        if hasattr(df[self.datetime_col].dt, "dayofyear"):
            df["day_of_year"] = df[self.datetime_col].dt.dayofyear
            self.logger.debug("Added 'day_of_year' feature.")

        # Add quarter
        if hasattr(df[self.datetime_col].dt, "quarter"):
            df["quarter"] = df[self.datetime_col].dt.quarter
            self.logger.debug("Added 'quarter' feature.")

        # Add hour if applicable
        if hasattr(df[self.datetime_col].dt, "hour"):
            if (df[self.datetime_col].dt.hour.nunique() > 1):  # Only add if hours are relevant
                df["hour"] = df[self.datetime_col].dt.hour
                self.logger.debug("Added 'hour' feature.")

        # Add minute if applicable
        if hasattr(df[self.datetime_col].dt, "minute"):
            if (df[self.datetime_col].dt.minute.nunique() > 1):  # Only add if minutes are relevant
                df["minute"] = df[self.datetime_col].dt.minute
                self.logger.debug("Added 'minute' feature.")

        # Add second if applicable
        if hasattr(df[self.datetime_col].dt, "second"):
            if (df[self.datetime_col].dt.second.nunique() > 1):  # Only add if seconds are relevant
                df["second"] = df[self.datetime_col].dt.second
                self.logger.debug("Added 'second' feature.")

        self.logger.info("Temporal features added successfully.")
        return df

    def _fourier_transform(
        self, series: np.ndarray, period: int = 365
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the FFT of a series and return frequencies and their magnitudes.
        """
        # Number of samples
        n = len(series)
        self.logger.info(f"Performing FFT on a series of length {n} with period {period}.")
        # Perform FFT
        fft_values = np.fft.fft(series)
        # Frequencies
        freqs = np.fft.fftfreq(n)
        # Magnitudes
        magnitudes = np.abs(fft_values)
        self.logger.debug(f"FFT completed. Found {len(freqs)} frequencies.")
        return freqs, magnitudes

    def _add_fourier_features(self, df: pd.DataFrame, period: int = 365):
        """
        Add top_k_fourier Fourier features to the DataFrame based on the target column.
        For simplicity, we'll do a global FFT on the entire series (not windowed).
        """
        self.logger.info(
            f"Adding top {self.top_k_fourier} Fourier components as features using column '{self.target_col}'."
        )
        target_series = df[self.target_col].values
        freqs, mags = self._fourier_transform(target_series, period=period)

        # We only consider positive frequencies
        positive_idx = np.where(freqs > 0)
        freqs = freqs[positive_idx]
        mags = mags[positive_idx]

        # Get top k by magnitude
        sorted_idx = np.argsort(mags)[::-1]
        top_k_idx = sorted_idx[: self.top_k_fourier]
        self.logger.debug(
            f"Top {self.top_k_fourier} frequencies selected: {freqs[top_k_idx]}."
        )

        for i, idx in enumerate(top_k_idx):
            # For each top frequency, add as a feature:
            feature_name = f"fft_freq_{i}"
            df[feature_name] = np.sin(
                2 * np.pi * freqs[idx] * np.arange(len(df))
            ) + np.cos(2 * np.pi * freqs[idx] * np.arange(len(df)))
            self.logger.debug(f"Fourier feature '{feature_name}' added.")
        self.logger.info("Fourier features added successfully.")
        return df

    def _create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        seq_length: int,
        forecast_horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences of length seq_length with the next forecast_horizon step as target.
        Typically used for supervised learning in time-series.
        """
        sequences_X = []
        sequences_y = []
        for i in range(len(X) - seq_length - forecast_horizon + 1):
            seq_x = X[i : i + seq_length]
            seq_y = y[i + seq_length + forecast_horizon - 1] if y is not None else None
            sequences_X.append(seq_x)
            if seq_y is not None:
                sequences_y.append(seq_y)

        self.logger.info(
            f"Generated {len(sequences_X)} sequences for features and targets."
        )

        sequences_X = np.array(sequences_X)
        if y is not None:
            sequences_y = np.array(sequences_y)
        else:
            sequences_y = None

        return sequences_X, sequences_y

    def fit_transform(
        self,
        df: pd.DataFrame,
        add_temporal: bool = True,
        add_fourier: bool = False,
        fourier_period: int = 365,
    ) -> pd.DataFrame:
        """
        Fit the transformations (encoders, scalers) on the given training DataFrame,
        then transform and return the processed DataFrame.
        """
        # Sort by datetime (important for time series)
        df = df.sort_values(by=self.datetime_col).reset_index(drop=True)
        # Remove NaNs if requested
        df = self._remove_nans(df)

        # Optional feature engineering
        if add_temporal:
            df = self._add_temporal_features(df)
        if add_fourier:
            df = self._add_fourier_features(df, period=fourier_period)

        # Encode categorical features
        df = self._encode_categorical(df, fit=True)

        # Save the final feature columns after encoding (excluding target)
        self.feature_columns_after_encoding = [
            col for col in df.columns if col not in [self.target_col, self.datetime_col]
        ]

        # Separate features and target
        X = df[self.feature_columns_after_encoding].values
        y = df[self.target_col].values

        # Apply scaling (fit, then transform)
        X_scaled, y_scaled = self._apply_scaling(X, y, fit=True)

        # Replace in DataFrame or just keep arrays
        processed_df = df.copy()
        processed_df[self.feature_columns_after_encoding] = X_scaled
        processed_df[self.target_col] = y_scaled if y_scaled is not None else y

        return processed_df

    def transform(
        self,
        df: pd.DataFrame,
        add_temporal: bool = True,
        add_fourier: bool = False,
        fourier_period: int = 365,
    ) -> pd.DataFrame:
        """
        Transform new data (inference or validation) using the previously fitted transformations.
        The new data might not have the target column if it's inference data.
        """
        df = df.sort_values(by=self.datetime_col).reset_index(drop=True)

        # If target_col not in df, we create a placeholder
        has_target = self.target_col in df.columns
        if not has_target:
            df[self.target_col] = 0.0  # dummy, will remove after scaling

        if self.remove_na:
            df = self._remove_nans(df)

        # Feature engineering
        if add_temporal:
            df = self._add_temporal_features(df)
        if add_fourier and has_target:
            # If the target is not available, this might not make sense,
            # but let's assume we skip if there's no target.
            df = self._add_fourier_features(df, period=fourier_period)

        # Encode categorical
        for cat_col in self.categorical_cols:
            if cat_col not in df.columns:
                df[cat_col] = ""  # in case the column is missing
        df = self._encode_categorical(df, fit=False)

        # Ensure columns match the training-encoded columns
        # If any columns are missing, add them with 0
        for col in self.feature_columns_after_encoding:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[
            [self.datetime_col]
            + self.feature_columns_after_encoding
            + [self.target_col]
        ]

        # Separate features and target
        X = df[self.feature_columns_after_encoding].values
        if has_target:
            y = df[self.target_col].values
        else:
            y = None

        # Apply scaling
        X_scaled, y_scaled = self._apply_scaling(X, y, fit=False)

        transformed_df = df.copy()
        transformed_df[self.feature_columns_after_encoding] = X_scaled
        if has_target and y_scaled is not None:
            transformed_df[self.target_col] = y_scaled

        if not has_target:
            transformed_df = transformed_df.drop(columns=[self.target_col])

        return transformed_df

    def prepare_for_prediction(
        self,
        df: pd.DataFrame,
        add_temporal: bool = True,
        add_fourier: bool = False,
        fourier_period: int = 365,
        return_dataloader: bool = False,
        batch_size: int = 32,
        shuffle: bool = False,
    ):
        """
        Prepare the test dataset for prediction by transforming it and creating sequences.
        Returns either numpy arrays or a PyTorch DataLoader.
        """
        # Transform the test data
        transformed_df = self.transform(
            df,
            add_temporal=add_temporal,
            add_fourier=add_fourier,
            fourier_period=fourier_period,
        )

        # Create sequences
        X_test = self._create_sequences(
            transformed_df[self.feature_columns_after_encoding].values,
            None,  # No target column for test data
            self.seq_length,
            self.forecast_horizon,
        )[
            0
        ]  # Only X, as y is None

        if not return_dataloader:
            return X_test
        else:
            # Create PyTorch TensorDataset
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=shuffle
            )
            return test_loader

    def split_data(
        self, df: pd.DataFrame, train_size: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data (already processed) into train and test sets based on the train_size ratio.
        """
        split_idx = int(len(df) * train_size)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        return train_df, test_df

    def get_arrays_or_loaders(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        return_dataloaders: bool = False,
        batch_size: int = 32,
        shuffle: bool = False,
    ):
        """
        Return either:
          - (X_train, X_test, y_train, y_test) as numpy arrays
          - or (train_loader, val_loader) as PyTorch Dataloaders
        """
        # Convert to sequences
        X_train, y_train = self._create_sequences(
            df_train[self.feature_columns_after_encoding].values,
            df_train[self.target_col].values,
            self.seq_length,
            self.forecast_horizon,
        )
        X_val, y_val = self._create_sequences(
            df_val[self.feature_columns_after_encoding].values,
            df_val[self.target_col].values,
            self.seq_length,
            self.forecast_horizon,
        )

        if not return_dataloaders:
            return X_train, X_val, y_train, y_val
        else:
            # Create PyTorch TensorDatasets
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader

    def plot_time_series(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        freq: str = "D",  # 'D' (daily), 'M' (monthly), 'Y' (yearly), etc.
        title: str = "Time Series Plot",
    ):
        """
        Plot the time series target between optional start and end dates.
        Also allows resampling by a given frequency.
        """
        # Ensure datetime col is datetime dtype
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        plot_df = df.copy()
        # Filter by date range if provided
        if start_date is not None:
            plot_df = plot_df[plot_df[self.datetime_col] >= start_date]
        if end_date is not None:
            plot_df = plot_df[plot_df[self.datetime_col] <= end_date]

        # Resample if freq is provided (for example: 'D', 'M', 'Y')
        plot_df.set_index(self.datetime_col, inplace=True)
        resampled_df = plot_df[self.target_col].resample(freq).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(
            resampled_df.index, resampled_df.values, marker="o", label=self.target_col
        )
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # -------------------------
    # Synthetic Data Generation
    # -------------------------
    np.random.seed(42)

    # Generate a date range (daily frequency)
    date_rng = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
    n_samples = len(date_rng)

    # Synthetic numeric features
    feature1 = np.random.randn(n_samples) * 10 + 100  # some random series
    feature2 = np.random.randn(n_samples) * 5 + 50

    # Synthetic categorical feature
    categories = ["A", "B", "C"]
    cat_feature = np.random.choice(categories, size=n_samples)

    # Synthetic target with some trend + seasonality
    t = np.arange(n_samples)
    target = (
        0.01 * t + 10 * np.sin(2 * np.pi * t / 365) + np.random.randn(n_samples) * 2
    )

    df_synthetic = pd.DataFrame(
        {
            "date": date_rng,
            "feature1": feature1,
            "feature2": feature2,
            "cat_feature": cat_feature,
            "target": target,
        }
    )

    # -------------------------
    # Instantiate Preprocessor
    # -------------------------
    preprocessor = TimeSeriesPreprocessor(
        datetime_col="date",
        target_col="target",
        categorical_cols=["cat_feature"],
        numeric_cols=["feature1", "feature2"],
        feature_scaler=StandardScaler(),
        target_scaler=MinMaxScaler(),
        seq_length=7,
        forecast_horizon=1,
        top_k_fourier=3,
    )

    # ---------------------------------
    # Fit Transform on Training Dataset
    # ---------------------------------
    processed_df = preprocessor.fit_transform(
        df_synthetic,
        add_temporal=True,
        add_fourier=True,  # add FFT-based features
        fourier_period=365,
    )

    # ------------------------------
    # Split into train / test
    # ------------------------------
    train_df, test_df = preprocessor.split_data(processed_df, train_size=0.8)

    # ---------------------------------------------------
    # Get either array-based splits or PyTorch DataLoaders
    # ---------------------------------------------------
    # 1) As arrays:
    X_train, X_val, y_train, y_val = preprocessor.get_arrays_or_loaders(
        train_df, test_df, return_dataloaders=False
    )
    print("Array shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)

    # 2) As PyTorch DataLoaders:
    train_loader, val_loader = preprocessor.get_arrays_or_loaders(
        train_df, test_df, return_dataloaders=True, batch_size=32, shuffle=True
    )

    print("\nDataLoader sizes (in terms of number of batches):")
    print("Number of train batches:", len(train_loader))
    print("Number of val batches:", len(val_loader))

    # -----------------------
    # Visualization Examples
    # -----------------------
    print("\nVisualizing the entire processed data (daily resolution):")
    preprocessor.plot_time_series(
        processed_df, freq="D", title="Entire Time Series (Daily)"
    )

    print("\nVisualizing zoomed-in data from 2020-06-01 to 2020-08-01:")
    preprocessor.plot_time_series(
        processed_df,
        start_date="2020-06-01",
        end_date="2020-08-01",
        freq="D",
        title="Zoomed-In Plot (Daily)",
    )

    print("\nVisualizing with monthly granularity:")
    preprocessor.plot_time_series(
        processed_df, freq="ME", title="Entire Time Series (Monthly)"
    )

    print("\nAll tests done.")
