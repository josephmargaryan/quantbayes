import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)


class CrossSectionalAnalysis:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        logging.info("Preprocessing data...")
        self.returns = self.data.pct_change().dropna()
        logging.info("Data preprocessed.")

    def apply_pca(self, n_components=2):
        logging.info("Applying PCA...")
        self.pca = PCA(n_components=n_components)
        self.pca_result = self.pca.fit_transform(self.returns.T)
        self.pca_explained_variance = self.pca.explained_variance_ratio_
        logging.info("PCA applied.")

    def plot_pca_results(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], color="blue")
        for i, stock in enumerate(self.returns.columns):
            plt.text(self.pca_result[i, 0], self.pca_result[i, 1], stock)
        plt.title("PCA Results for Stock Returns")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        plt.show()

    def plot_explained_variance(self):
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(1, len(self.pca_explained_variance) + 1),
            self.pca_explained_variance,
            color="orange",
        )
        plt.title("Explained Variance by Principal Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    import yfinance as yf

    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX",
        "ADBE",
        "ORCL",
        "PYPL",
        "CRM",
        "INTC",
        "CSCO",
        "AMD",
        "IBM",
        "QCOM",
        "TXN",
        "AVGO",
        "NOW",
        "SNOW",
        "SHOP",
        "SQ",
        "ZM",
        "UBER",
        "LYFT",
        "TWLO",
        "ROKU",
        "DOCU",
        "PINS",
    ]
    data = pd.DataFrame()
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = stock.history(start="2020-01-01", interval="1d")["Close"]

    # Cross-Sectional Analysis
    analysis = CrossSectionalAnalysis(data)
    analysis.preprocess_data()
    analysis.apply_pca(n_components=2)
    analysis.plot_pca_results()
    analysis.plot_explained_variance()
