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
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    
    # Moving Averages
    df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    
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
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        df = load_stock(symbol)
        if df is None:
            continue
        df = add_features(df)
        df = create_target(df)
        df.dropna(inplace=True)
        df.to_csv(f"data/processed/{symbol}.csv", index=False)
        print(f"[OK] {symbol} processed — {len(df)} rows")

if __name__ == "__main__":
    process_all()
