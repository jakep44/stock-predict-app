import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
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
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

df = add_features(df)

# Prepare data
features = ['Return', 'SMA_10', 'SMA_50', 'Volume_Change']
X = df[features]
y = df['Target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
accuracy = (predictions == y_test).mean()

# Predict next movement
latest_features = df[features].iloc[-1].values.reshape(1, -1)
next_prediction = model.predict(latest_features)[0]
movement = "UP 📈" if next_prediction == 1 else "DOWN 📉"

# Display chart
st.subheader(f"{ticker} Price Chart with Moving Averages")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Candles"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['SMA_10'],
    mode='lines',
    name='SMA 10',
    line=dict(color='orange')
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['SMA_50'],
    mode='lines',
    name='SMA 50',
    line=dict(color='blue')
))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Show prediction & accuracy
st.subheader(f"Model Accuracy: {accuracy:.2%}")
st.subheader(f"Predicted Next Hourly Movement: {movement}")

# Show raw data
with st.expander("Show raw data"):
    st.dataframe(df)
