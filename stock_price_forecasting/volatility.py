import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# Fetch data
def fetch_data(ticker, start_date, interval="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, interval=interval)
    return data[["High", "Low"]]


# Calculate High-Low Range
def calculate_high_low_range(data):
    data["High-Low Range"] = data["High"] - data["Low"]
    return data


# Plot High-Low Range
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


# Main
if __name__ == "__main__":
    ticker = "AAPL"  # Replace with your desired stock ticker
    start_date = "2023-01-01"

    data = fetch_data(ticker, start_date)
    data = calculate_high_low_range(data)
    print(data.head())  # Display sample data

    plot_high_low_range(data, ticker)
