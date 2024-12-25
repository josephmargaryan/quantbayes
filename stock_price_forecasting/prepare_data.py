from get_data import StockData
import pandas as pd


sentiment_data = pd.read_csv("data.csv")

company_stock_data = {}

unique_companies = sentiment_data["Company"].unique()

start_date = "2023-01-01"

for company in unique_companies:
    try:
        stock_instance = StockData(ticker=company, start_date=start_date, interval="1d")
        if stock_instance.data is not None:
            stock_instance.data["Company"] = company
            company_stock_data[company] = stock_instance.data.reset_index()
    except Exception as e:
        print(f"Error processing data for {company}: {e}")

stock_data_combined = pd.concat(company_stock_data.values(), ignore_index=True)

merged_data = pd.merge(sentiment_data, stock_data_combined, on="Company", how="inner")
merged_data = merged_data[
    ["Company", "predictions", "Close", "Volatility", "RSI"]
].rename(columns={"predictions": "predicted_labels"})
merged_data.to_csv("merged_data.csv", index=False)
print("Merged data saved to 'merged_data.csv'")
