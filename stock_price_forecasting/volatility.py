import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def fetch_data(ticker, start_date, interval="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, interval=interval)
    return data[["High", "Low"]]


def calculate_high_low_range(data):
    data["High-Low Range"] = data["High"] - data["Low"]
    return data


def plot_high_low_range(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.index,
        data["High-Low Range"],
        label="High-Low Range",
        color="orange",
        linewidth=2,
    )
    plt.title(f"{ticker} High-Low Range Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price Range")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    ticker = "AAPL"  
    start_date = "2023-01-01"

    data = fetch_data(ticker, start_date)
    data = calculate_high_low_range(data)
    print(data.head())  

    plot_high_low_range(data, ticker)
