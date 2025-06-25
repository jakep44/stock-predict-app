import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 PredictTick – Stock Movement with Advanced Liquidity Sweeps")

# Sidebar input
ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL, TSLA)", "AAPL")
raw_interval = st.sidebar.selectbox("Raw Data Interval", ["30m", "1h"], index=1)
lookback_days = st.sidebar.selectbox("How many days of raw data", ["30d", "60d", "90d"], index=1)

# Load raw high-res data
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

df_raw = load_data(ticker, lookback_days, raw_interval)

# Ensure index is datetime for resampling
df_raw.index = pd.to_datetime(df_raw.index)

# Flatten multi-level columns if needed (yfinance sometimes does this)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = [' '.join(col).strip() for col in df_raw.columns.values]

st.write("Available columns after flattening:", df_raw.columns.tolist())

# Add features
def add_features(df):
    # Find correct 'Close' and 'Volume' columns
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

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict next move
latest_features = df_raw[features].iloc[-1].values.reshape(1, -1)
next_prediction = model.predict(latest_features)[0]
pred_proba = model.predict_proba(latest_features)[0][1]
movement = "UP 📈" if next_prediction == 1 else "DOWN 📉"
confidence = f"{pred_proba*100:.1f}% Confidence"

# Aggregate to daily for chart
required_cols = [col for col in df_raw.columns if any(x in col for x in ['Open', 'High', 'Low', 'Close', 'Volume'])]

# Extract proper columns for resample
col_map = {}
for name in ['Open', 'High', 'Low', 'Close', 'Volume']:
    col_found = next((col for col in df_raw.columns if name in col), None)
    if col_found:
        col_map[name] = col_found

if len(col_map) < 5:
    st.error(f"Missing columns for resampling: {col_map}")
else:
    df_daily = df_raw.resample('1D').agg({
        col_map['Open']: 'first',
        col_map['High']: 'max',
        col_map['Low']: 'min',
        col_map['Close']: 'last',
        col_map['Volume']: 'sum'
    })
    df_daily.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_daily.dropna(inplace=True)

    # Detect liquidity sweeps (mark last 5 occurrences)
    df_daily['Prev_High'] = df_daily['High'].shift(1)
    df_daily['Prev_Low'] = df_daily['Low'].shift(1)
    df_daily['Sweep_Up'] = (df_daily['High'] > df_daily['Prev_High'])
    df_daily['Sweep_Down'] = (df_daily['Low'] < df_daily['Prev_Low'])

    sweep_up = df_daily[df_daily['Sweep_Up']].tail(5)
    sweep_down = df_daily[df_daily['Sweep_Down']].tail(5)

    # Price chart with sweeps, RSI, prediction line
    st.subheader(f"{ticker} 1-Day Price Chart with Advanced Liquidity Sweeps and RSI")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_daily.index,
        open=df_daily['Open'],
        high=df_daily['High'],
        low=df_daily['Low'],
        close=df_daily['Close'],
        name="Daily Candles"
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

    future_x = [df_daily.index[-1] + pd.Timedelta(days=1)]
    future_y = [df_daily['Close'].iloc[-1] * (1.01 if next_prediction == 1 else 0.99)]

    fig.add_trace(go.Scatter(
        x=future_x,
        y=future_y,
        mode='lines+markers',
        line=dict(color='blue', dash='dot'),
        marker=dict(size=14),
        name=f'Predicted Move ({movement})'
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# RSI Chart
st.subheader("Relative Strength Index (RSI)")
st.line_chart(df_raw[['RSI']].dropna())

# Show results
accuracy = (model.predict(X_test) == y_test).mean()
st.subheader(f"Model Accuracy: {accuracy:.2%}")
st.subheader(f"Predicted Next Movement: {movement} with {confidence}")

with st.expander("Show raw data"):
    st.dataframe(df_raw.tail(100))
