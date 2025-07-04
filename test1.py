import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 PredictTick – Real-Time Day Chart with Liquidity Sweeps and Prediction")

# Sidebar input
ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL, TSLA)", "AAPL")
raw_interval = st.sidebar.selectbox("Raw Data Interval", ["1m", "5m", "15m"], index=1)
lookback_days = st.sidebar.selectbox("How many days of raw data", ["1d", "5d", "10d", "30d"], index=3)

# Load raw intraday data
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

df_raw = load_data(ticker, lookback_days, raw_interval)

# Print columns for debugging
st.write("Raw Data Columns:", df_raw.columns.tolist())

# Standardize columns if possible
expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
mapping = {col: str(col).split()[-1] for col in df_raw.columns if any(key in str(col) for key in expected_cols)}
df_raw.rename(columns=mapping, inplace=True)

# Add features
def add_features(df):
    close_col = next((col for col in df.columns if 'Close' in col), None)
    volume_col = next((col for col in df.columns if 'Volume' in col), None)

    if not close_col or not volume_col:
        st.error(f"Couldn't locate 'Close' or 'Volume' columns in: {df.columns.tolist()}")
        return df

    df['Return'] = df[close_col].pct_change()
    df['SMA_10'] = df[close_col].rolling(window=10).mean()
    df['SMA_50'] = df[close_col].rolling(window=50).mean()
    df['Volume_Change'] = df[volume_col].pct_change()
    df['RSI'] = compute_rsi(df[close_col], window=14)
    df['Target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)
    avg_gain = gain_series.rolling(window=window).mean()
    avg_loss = loss_series.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_raw = add_features(df_raw)

# Train model
features = ['Return', 'SMA_10', 'SMA_50', 'Volume_Change', 'RSI']
X = df_raw[features]
y = df_raw['Target']

if len(X) < 10:
    st.error("Not enough data to train the model. Try selecting a longer time period or different interval.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    latest_features = X.iloc[-1].values.reshape(1, -1)
    next_prediction = model.predict(latest_features)[0]
    pred_proba = model.predict_proba(latest_features)[0][1]
    movement = "UP 📈" if next_prediction == 1 else "DOWN 📉"
    confidence = f"{pred_proba*100:.1f}% Confidence"

    # Liquidity sweeps based on intraday highs/lows
    high_col = next((col for col in df_raw.columns if 'High' in col), None)
    low_col = next((col for col in df_raw.columns if 'Low' in col), None)

    if not high_col or not low_col:
        st.error(f"Couldn't locate 'High' or 'Low' columns in: {df_raw.columns.tolist()}")
    else:
        df_raw['Prev_High'] = df_raw[high_col].shift(1)
        df_raw['Prev_Low'] = df_raw[low_col].shift(1)
        df_raw['Sweep_Up'] = (df_raw[high_col] > df_raw['Prev_High'])
        df_raw['Sweep_Down'] = (df_raw[low_col] < df_raw['Prev_Low'])
        
        sweep_up = df_raw[df_raw['Sweep_Up']].tail(5)
        sweep_down = df_raw[df_raw['Sweep_Down']].tail(5)

        st.subheader(f"{ticker} Live Day Chart with Liquidity Sweeps and Prediction")
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_raw.index,
            open=df_raw['Open'],
            high=df_raw[high_col],
            low=df_raw[low_col],
            close=df_raw['Close'],
            name="Intraday Candles"
        ))

        fig.add_trace(go.Scatter(
            x=sweep_up.index,
            y=sweep_up[high_col],
            mode='markers',
            marker=dict(color='green', size=12, symbol='triangle-up'),
            name='Liquidity Sweep Up'
        ))

        fig.add_trace(go.Scatter(
            x=sweep_down.index,
            y=sweep_down[low_col],
            mode='markers',
            marker=dict(color='red', size=12, symbol='triangle-down'),
            name='Liquidity Sweep Down'
        ))

        # Predicted continuation - yellow projected price point
        future_x = [df_raw.index[-1] + pd.Timedelta(minutes=5)]
        future_y = [df_raw['Close'].iloc[-1] * (1.01 if next_prediction == 1 else 0.99)]

        fig.add_trace(go.Scatter(
            x=future_x,
            y=future_y,
            mode='markers+lines',
            line=dict(color='yellow', dash='dot'),
            marker=dict(size=14, color='yellow'),
            name=f'Predicted Continuation ({movement})'
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # RSI Chart
        st.subheader("Relative Strength Index (RSI)")

# Flatten columns if MultiIndex
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = [' '.join(map(str, col)).strip() for col in df_raw.columns]

if 'RSI' in df_raw.columns:
    st.line_chart(df_raw[['RSI']].dropna())
else:
    st.warning("RSI column not found. Please verify data source.")

