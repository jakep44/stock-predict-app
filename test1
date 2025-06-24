import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("📈 PredictTick – Stock Movement Predictor")

# Sidebar input
ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL, TSLA)", "AAPL")
period = st.sidebar.selectbox("Data range", ["30d", "60d", "90d"], index=1)
interval = st.sidebar.selectbox("Interval", ["30m", "1h", "1d"], index=1)

# Load data
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

df = load_data(ticker, period, interval)

# Add features
def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Target'] = (df['Close'].sh
