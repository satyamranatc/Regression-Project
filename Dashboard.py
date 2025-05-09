import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from Api import GetData  # Assuming this is your custom API module

# List of available companies
companies = [
    "AAPL",  # Apple
    "TSLA",  # Tesla
    "IBM",   # IBM
    "GOOGL", # Alphabet (Google)
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta (Facebook)
    "NFLX",  # Netflix
    "NVDA",  # NVIDIA
    "ADBE"   # Adobe
]


def FetchData(companies):
    for company in companies:
        if GetData(company):
            pass
            # st.success(f"Data Stored For {company} Successfully!")


st.sidebar.header("Select Company:")
selected_company = st.sidebar.selectbox("Select a company", companies)

if st.sidebar.button("Fetch and Store Data"):
    FetchData(companies)
    st.success("Data Fetched and Stored Successfully!")


# App title and header
st.title(f"📈 Forex Stock Analytics {selected_company}")
st.write("### Get the Latest Data:")


# Read and display stored CSV data
data_dfs = []
data_folder = "./Data"

for file_name in os.listdir(data_folder):
    print(file_name)
    print(selected_company)
    file_path = os.path.join(data_folder, file_name)
    if file_name.split("_")[0] == selected_company:
        df = pd.read_csv(file_path)
        st.dataframe(df)
        data_dfs.append({"Company": file_name, "data": df})

        # Ensure timestamp is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'])])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add 5 important stock market calculations
        st.header("Key Stock Market Metrics")
        
        # Calculate metrics
        if len(df) > 0:
            # 1. Moving Averages
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # 2. Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. Bollinger Bands
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['STD_20'] = df['close'].rolling(window=20).std()
            df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
            df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)
            
            # 4. MACD (Moving Average Convergence Divergence)
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # 5. Average True Range (ATR) - Volatility Indicator
            df['High-Low'] = df['high'] - df['low']
            df['High-PrevClose'] = abs(df['high'] - df['close'].shift(1))
            df['Low-PrevClose'] = abs(df['low'] - df['close'].shift(1))
            df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Moving Averages")
                st.write(f"20-Day SMA: ${df['SMA_20'].iloc[-1]:.2f}")
                st.write(f"50-Day SMA: ${df['SMA_50'].iloc[-1]:.2f}")
                st.write(f"Current Price: ${df['close'].iloc[-1]:.2f}")
                st.write(f"Price to 50-Day SMA Ratio: {(df['close'].iloc[-1] / df['SMA_50'].iloc[-1]):.2f}")
                
                st.subheader("Bollinger Bands")
                st.write(f"Upper Band: ${df['Upper_Band'].iloc[-1]:.2f}")
                st.write(f"Middle Band (20-SMA): ${df['SMA_20'].iloc[-1]:.2f}")
                st.write(f"Lower Band: ${df['Lower_Band'].iloc[-1]:.2f}")
                
            with col2:
                st.subheader("RSI (14)")
                rsi_value = df['RSI'].iloc[-1]
                st.write(f"RSI Value: {rsi_value:.2f}")
                
                # RSI interpretation
                if rsi_value < 30:
                    st.write("Status: Potentially Oversold")
                elif rsi_value > 70:
                    st.write("Status: Potentially Overbought")
                else:
                    st.write("Status: Neutral")
                
                st.subheader("MACD")
                st.write(f"MACD Line: {df['MACD'].iloc[-1]:.4f}")
                st.write(f"Signal Line: {df['MACD_Signal'].iloc[-1]:.4f}")
                st.write(f"Histogram: {df['MACD_Histogram'].iloc[-1]:.4f}")
                
                st.subheader("Volatility (ATR-14)")
                st.write(f"ATR Value: ${df['ATR'].iloc[-1]:.2f}")
                st.write(f"ATR as % of price: {(df['ATR'].iloc[-1] / df['close'].iloc[-1] * 100):.2f}%")
            
            # Create visual for trend analysis
            st.header("Technical Analysis Visualization")
            
            # Plot Moving Averages with Price
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Price', line=dict(color='black')))
            fig_ma.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_20'], name='20-Day MA', line=dict(color='blue')))
            fig_ma.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_50'], name='50-Day MA', line=dict(color='red')))
            fig_ma.update_layout(title='Price vs Moving Averages', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_ma, use_container_width=True)
            
            # Display recent price change
            if len(df) >= 2:
                recent_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                if recent_change > 0:
                    st.success(f"Recent Change: +{recent_change:.2f}%")
                else:
                    st.error(f"Recent Change: {recent_change:.2f}%")
                
            # Overall trend analysis
            st.subheader("Trend Analysis")
            
            # Determine short-term trend (based on price vs 20-day SMA)
            if df['close'].iloc[-1] > df['SMA_20'].iloc[-1]:
                short_term = "Bullish"
            else:
                short_term = "Bearish"
                
            # Determine long-term trend (based on price vs 50-day SMA)
            if df['close'].iloc[-1] > df['SMA_50'].iloc[-1]:
                long_term = "Bullish"
            else:
                long_term = "Bearish"
                
            st.write(f"Short-term trend (vs 20-Day MA): **{short_term}**")
            st.write(f"Long-term trend (vs 50-Day MA): **{long_term}**")