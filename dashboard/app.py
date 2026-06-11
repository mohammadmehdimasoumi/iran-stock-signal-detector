import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SYMBOLS = [
    "Foolad", "Khodro", "Shasta", "Vbmelat", "Fars",
    "Zob", "Kegel", "Shabdar", "Vasandogh", "Simorgh"
]

FEATURES = [
    "rsi", "macd", "macd_signal",
    "bb_upper", "bb_lower",
    "sma_20", "sma_50",
    "volume_ratio", "momentum_5", "momentum_10",
    "prev_day_change", "dist_from_high", "dist_from_low",
    "price_to_sma20", "price_to_sma50"
]

@st.cache_data
def load_data():
    dfs = []
    for symbol in SYMBOLS:
        path = f"data/processed/{symbol}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["symbol"] = symbol
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=4
    )
    model.fit(X_train, y_train)
    return model

st.title("📈 Iran Stock Signal Detector")
st.caption("ML-based buy signal detection for Tehran Stock Exchange")

df = load_data()
model = train_model(df)

# سیگنال‌های امروز
st.subheader("🔔 Latest Signals")
latest = []
for symbol in SYMBOLS:
    path = f"data/processed/{symbol}.csv"
    if os.path.exists(path):
        sdf = pd.read_csv(path)
        last_row = sdf[FEATURES].iloc[[-1]]
        prob = model.predict_proba(last_row)[0][1]
        latest.append({"Symbol": symbol, "Buy Probability": round(prob, 2)})

signals_df = pd.DataFrame(latest).sort_values("Buy Probability", ascending=False)
st.dataframe(signals_df, use_container_width=True)

# نمودار قیمت
st.subheader("📊 Price Chart")
selected = st.selectbox("Select Stock", SYMBOLS)
sdf = pd.read_csv(f"data/processed/{selected}.csv")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sdf["close"].values, label="Close Price")
ax.set_title(f"{selected} - Close Price")
ax.legend()
st.pyplot(fig)

# RSI
st.subheader("📉 RSI")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(sdf["rsi"].values, color="orange", label="RSI")
ax2.axhline(70, color="red", linestyle="--")
ax2.axhline(30, color="green", linestyle="--")
ax2.legend()
st.pyplot(fig2)
