import requests
import pandas as pd
import matplotlib.pyplot as plt
import logging
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)


class StockSemanticAnalysis:
    def __init__(self, tickers):
        self.tickers = tickers
        self.sentiment_scores = {}

    def fetch_news(self, ticker):
        logging.info(f"Fetching news for {ticker}...")
        # Simulated API call - replace with a real news API like NewsAPI or Alpha Vantage
        api_url = f"https://newsapi.org/v2/everything?q={ticker}&from=2024-12-20&to=2024-12-22&sortBy=publishedAt&apiKey=ec402d1a09cc42cab898607cad30a101"
        response = requests.get(api_url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return [
                article["title"] + " " + article["description"]
                for article in articles
                if article["title"] and article["description"]
            ]
        else:
            logging.warning(f"Failed to fetch news for {ticker}")
            return []

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def process_ticker(self, ticker):
        news_articles = self.fetch_news(ticker)
        if not news_articles:
            self.sentiment_scores[ticker] = None
            return

        sentiments = [self.analyze_sentiment(article) for article in news_articles]
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        self.sentiment_scores[ticker] = average_sentiment

    def analyze_stocks(self):
        logging.info("Analyzing stock sentiments...")
        for ticker in self.tickers:
            self.process_ticker(ticker)
        logging.info("Sentiment analysis complete.")

    def classify_sentiments(self):
        self.sentiment_classification = {
            "Positive": [
                ticker
                for ticker, score in self.sentiment_scores.items()
                if score is not None and score > 0.1
            ],
            "Neutral": [
                ticker
                for ticker, score in self.sentiment_scores.items()
                if score is not None and -0.1 <= score <= 0.1
            ],
            "Negative": [
                ticker
                for ticker, score in self.sentiment_scores.items()
                if score is not None and score < -0.1
            ],
        }

    def display_results(self):
        for sentiment, tickers in self.sentiment_classification.items():
            print(f"{sentiment} Sentiment:")
            print(", ".join(tickers) if tickers else "None")

    def plot_sentiments(self):
        categories = ["Positive", "Neutral", "Negative"]
        counts = [
            len(self.sentiment_classification[category]) for category in categories
        ]

        plt.figure(figsize=(8, 5))
        plt.bar(categories, counts, color=["green", "gray", "red"])
        plt.title("Stock Sentiment Classification")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Stocks")
        plt.grid(axis="y")
        plt.show()

    def plot_sentiment_per_stock(self):
        stocks = list(self.sentiment_scores.keys())
        sentiments = [
            (
                self.sentiment_scores[stock]
                if self.sentiment_scores[stock] is not None
                else 0
            )
            for stock in stocks
        ]

        plt.figure(figsize=(12, 8))
        colors = [
            "green" if score > 0.1 else "gray" if -0.1 <= score <= 0.1 else "red"
            for score in sentiments
        ]
        plt.bar(stocks, sentiments, color=colors)
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.title("Sentiment Scores per Stock")
        plt.xlabel("Stocks")
        plt.ylabel("Sentiment Score")
        plt.xticks(rotation=90)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
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

    analysis = StockSemanticAnalysis(tickers)
    analysis.analyze_stocks()
    analysis.classify_sentiments()
    analysis.display_results()
    analysis.plot_sentiments()
    analysis.plot_sentiment_per_stock()
