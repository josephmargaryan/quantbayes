import time
import feedparser
from bs4 import BeautifulSoup
import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


class NewsFetcher:
    def __init__(self, tickers, delay=1):
        """Initialize the NewsFetcher class.

        Args:
            tickers (dict): A dictionary mapping company tickers to their RSS feed URLs.
            delay (int): Delay in seconds between requests to avoid rate limits.
        """
        self.tickers = tickers
        self.delay = delay

    def fetch_rss_news(self, feed_url):
        """Fetch news using RSS feed.

        Args:
            feed_url (str): URL of the RSS feed.

        Returns:
            list: List of news articles fetched from the RSS feed.
        """
        logging.info(f"Fetching RSS news from {feed_url}...")
        try:
            feed = feedparser.parse(feed_url)
            return [
                entry["title"] + " " + entry.get("summary", "")
                for entry in feed.entries
            ]
        except Exception as e:
            logging.error(f"Error fetching RSS news: {e}")
            return []

    def fetch_google_news(self, ticker):
        """Fetch news using Google News Search.

        Args:
            ticker (str): Stock ticker symbol to search for.

        Returns:
            list: List of news articles fetched from Google News.
        """
        logging.info(f"Fetching Google News for {ticker}...")
        url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
            return [article.text for article in articles]
        except Exception as e:
            logging.error(f"Error fetching Google News: {e}")
            return []

    def fetch_combined_news(self, ticker, rss_url=None, sources="both"):
        """Fetch combined news using RSS and/or Google News.

        Args:
            ticker (str): Stock ticker symbol.
            rss_url (str): URL of the RSS feed for the ticker.
            sources (str): "rss", "google", or "both" to specify sources to use.

        Returns:
            list: Combined list of news articles.
        """
        news = []

        if sources in ["rss", "both"] and rss_url:
            news.extend(self.fetch_rss_news(rss_url))

        if sources in ["google", "both"]:
            news.extend(self.fetch_google_news(ticker))

        return news

    def fetch_news_for_all(self, sources="both"):
        """Fetch news for all tickers.

        Args:
            sources (str): "rss", "google", or "both" to specify sources to use.

        Returns:
            pd.DataFrame: DataFrame containing unique news articles and associated companies.
        """
        logging.info("Fetching news for all tickers...")
        all_news = []

        for ticker, rss_url in self.tickers.items():
            articles = self.fetch_combined_news(ticker, rss_url, sources)
            all_news.extend(
                [{"Company": ticker, "News": article} for article in articles]
            )
            logging.info(f"Fetched {len(articles)} articles for {ticker}.")
            time.sleep(self.delay)  # Add delay between requests

        # Deduplicate news
        logging.info("Removing duplicate articles...")
        df = pd.DataFrame(all_news)
        df = df.drop_duplicates(subset=["News"]).reset_index(drop=True)

        logging.info("News fetching complete.")
        return df


if __name__ == "__main__":
    tickers = {
        # Technology
        "AAPL": "https://finance.yahoo.com/rss/headline?s=AAPL",
        "MSFT": "https://finance.yahoo.com/rss/headline?s=MSFT",
        "GOOGL": "https://finance.yahoo.com/rss/headline?s=GOOGL",
        "AMZN": "https://finance.yahoo.com/rss/headline?s=AMZN",
        "TSLA": "https://finance.yahoo.com/rss/headline?s=TSLA",
        "NVDA": "https://finance.yahoo.com/rss/headline?s=NVDA",
        "META": "https://finance.yahoo.com/rss/headline?s=META",
        "ADBE": "https://finance.yahoo.com/rss/headline?s=ADBE",
        "INTC": "https://finance.yahoo.com/rss/headline?s=INTC",
        "CRM": "https://finance.yahoo.com/rss/headline?s=CRM",
        # Healthcare
        "JNJ": "https://finance.yahoo.com/rss/headline?s=JNJ",
        "PFE": "https://finance.yahoo.com/rss/headline?s=PFE",
        "MRNA": "https://finance.yahoo.com/rss/headline?s=MRNA",
        "LLY": "https://finance.yahoo.com/rss/headline?s=LLY",
        "BNTX": "https://finance.yahoo.com/rss/headline?s=BNTX",
        # Financials
        "JPM": "https://finance.yahoo.com/rss/headline?s=JPM",
        "BAC": "https://finance.yahoo.com/rss/headline?s=BAC",
        "GS": "https://finance.yahoo.com/rss/headline?s=GS",
        "C": "https://finance.yahoo.com/rss/headline?s=C",
        "WFC": "https://finance.yahoo.com/rss/headline?s=WFC",
        # Energy
        "XOM": "https://finance.yahoo.com/rss/headline?s=XOM",
        "CVX": "https://finance.yahoo.com/rss/headline?s=CVX",
        "BP": "https://finance.yahoo.com/rss/headline?s=BP",
        "COP": "https://finance.yahoo.com/rss/headline?s=COP",
        "TOT": "https://finance.yahoo.com/rss/headline?s=TOT",
        # Consumer Goods
        "PG": "https://finance.yahoo.com/rss/headline?s=PG",
        "KO": "https://finance.yahoo.com/rss/headline?s=KO",
        "PEP": "https://finance.yahoo.com/rss/headline?s=PEP",
        "UL": "https://finance.yahoo.com/rss/headline?s=UL",
        "MCD": "https://finance.yahoo.com/rss/headline?s=MCD",
        # Industrials
        "CAT": "https://finance.yahoo.com/rss/headline?s=CAT",
        "BA": "https://finance.yahoo.com/rss/headline?s=BA",
        "GE": "https://finance.yahoo.com/rss/headline?s=GE",
        "LMT": "https://finance.yahoo.com/rss/headline?s=LMT",
        "UPS": "https://finance.yahoo.com/rss/headline?s=UPS",
        # Retail
        "WMT": "https://finance.yahoo.com/rss/headline?s=WMT",
        "TGT": "https://finance.yahoo.com/rss/headline?s=TGT",
        "COST": "https://finance.yahoo.com/rss/headline?s=COST",
        "HD": "https://finance.yahoo.com/rss/headline?s=HD",
        "LOW": "https://finance.yahoo.com/rss/headline?s=LOW",
        # Telecommunications
        "VZ": "https://finance.yahoo.com/rss/headline?s=VZ",
        "T": "https://finance.yahoo.com/rss/headline?s=T",
        "TMUS": "https://finance.yahoo.com/rss/headline?s=TMUS",
        # Airlines
        "DAL": "https://finance.yahoo.com/rss/headline?s=DAL",
        "AAL": "https://finance.yahoo.com/rss/headline?s=AAL",
        "UAL": "https://finance.yahoo.com/rss/headline?s=UAL",
        # Automobiles
        "F": "https://finance.yahoo.com/rss/headline?s=F",
        "GM": "https://finance.yahoo.com/rss/headline?s=GM",
        "RIVN": "https://finance.yahoo.com/rss/headline?s=RIVN",
    }

    # Initialize NewsFetcher with 2-second delay
    news_fetcher = NewsFetcher(tickers, delay=2)

    # Fetch news for all tickers
    news_df = news_fetcher.fetch_news_for_all(sources="both")

    # Save the latest news to a CSV file
    news_df.to_csv("latest_news.csv", index=False)
    print("News saved to 'latest_news.csv'.")
