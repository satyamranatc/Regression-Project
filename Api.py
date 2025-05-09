import requests
import pandas as pd
import os

def GetData(Company="IBM"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={Company}&interval=5min&apikey=O6DMM9OMXYOG6E9V"
    res = requests.get(url)
    data = res.json()

    # Check if data has the key
    if "Time Series (5min)" in data:
        time_series = data["Time Series (5min)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')

        # Clean column names: remove numbering like '1. open'
        df.columns = [col.split(' ', 1)[1] for col in df.columns]

        # Reset index so that timestamp becomes a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)

        # Make sure Data folder exists
        os.makedirs("./Data", exist_ok=True)

        print("-----------------------")
        # print(df.head())  # Now it will show timestamp + columns

        # Save to CSV
        df.to_csv(f"./Data/{Company}_Data.csv", index=False)

        return True
    else:
        print("Error or limit reached. Response:", data)
        return None

if __name__ == "__main__":
    success = GetData("TSLA")
    print("Data fetched and saved!" if success else "Failed to fetch data.")
