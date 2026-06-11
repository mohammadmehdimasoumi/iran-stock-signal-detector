import pandas as pd
import ta
import os

def load_stock(eng_name):
    path = f"data/raw/{eng_name}.csv"
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def add_features(df):

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()

    # --- فیچرهای جدید ---
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["prev_day_change"] = df["close"].pct_change(1)

    df["high_52w"] = df["close"].rolling(252).max()
    df["low_52w"] = df["close"].rolling(252).min()
    df["dist_from_high"] = (df["close"] - df["high_52w"]) / df["high_52w"]
    df["dist_from_low"] = (df["close"] - df["low_52w"]) / df["low_52w"]

    df["price_to_sma20"] = df["close"] / df["sma_20"]
    df["price_to_sma50"] = df["close"] / df["sma_50"]

    return df

def create_target(df, days=5, threshold=0.05):
    df["future_return"] = df["close"].shift(-days) / df["close"] - 1
    df["signal"] = (df["future_return"] > threshold).astype(int)
    return df

def process_all():
    os.makedirs("data/processed", exist_ok=True)

    symbols = [
        "Foolad", "Khodro", "Shasta", "Vbmelat", "Fars",
        "Zob", "Kegel", "Shabdar", "Vasandogh", "Simorgh"
    ]

    for eng_name in symbols:
        print(f"Processing {eng_name}...")
        df = load_stock(eng_name)
        if df is None:
            continue
        df = add_features(df)
        df = create_target(df)
        df.dropna(inplace=True)
        df.to_csv(f"data/processed/{eng_name}.csv", index=False)
        print(f"[OK] {eng_name} processed — {len(df)} rows")

if __name__ == "__main__":
    process_all()
