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
lookback_days = st.sidebar.selectbox("How many days of raw data", ["1d"], index=0)

# Load raw intraday data
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

df_raw = load_data(ticker, lookback_days, raw_interval)
df_raw.index = pd.to_datetime(df_raw.index)
df_raw.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df_raw.columns]

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
    df_raw['Prev_High'] = df_raw['High'].shift(1)
    df_raw['Prev_Low'] = df_raw['Low'].shift(1)
    df_raw['Sweep_Up'] = (df_raw['High'] > df_raw['Prev_High'])
    df_raw['Sweep_Down'] = (df_raw['Low'] < df_raw['Prev_Low'])

    sweep_up = df_raw[df_raw['Sweep_Up']].tail(5)
    sweep_down = df_raw[df_raw['Sweep_Down']].tail(5)

    st.subheader(f"{ticker} Live Day Chart with Liquidity Sweeps and Prediction")
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_raw.index,
        open=df_raw['Open'],
        high=df_raw['High'],
        low=df_raw['Low'],
        close=df_raw['Close'],
        name="Intraday Candles"
    ))

    fig.add_trace(go.Scatter(
        x=sweep_up.index,
        y=sweep_up['High'],
        mode='markers',
        marker=dict(color='green', size=12, symbol='triangle-up'),
        name='Liquidity Sweep Up'
    ))

    fig.add_trace(go.Scatter(
        x=sweep_down.index,
        y=sweep_down['Low'],
        mode='markers',
        marker=dict(color='red', size=12, symbol='triangle-down'),
        name='Liquidity Sweep Down'
    ))

    # Predicted continuation - yellow projected price point
    future_x = [df_raw.index[-1] + pd.Timedelta(minutes=5)]
    future_y = [df_raw['Close'].iloc[-1] * (1.005 if next_prediction == 1 else 0.995)]

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
    st.line_chart(df_raw[['RSI']].dropna())

    st.subheader(f"Model Accuracy: {(model.predict(X_test) == y_test).mean():.2%}")
    st.subheader(f"Predicted Next Movement: {movement} with {confidence}")

    with st.expander("Show raw data"):
        st.dataframe(df_raw.tail(100))
