import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
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

def load_all_data():
    dfs = []
    for symbol in SYMBOLS:
        path = f"data/processed/{symbol}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["symbol"] = symbol
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def train_models(df):
    X = df[FEATURES]
    y = df["signal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(classification_report(y_test, rf_preds))
    print(f"Precision: {precision_score(y_test, rf_preds):.2f}")

    # XGBoost
    print("\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=4
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    print(classification_report(y_test, xgb_preds))
    print(f"Precision: {precision_score(y_test, xgb_preds):.2f}")

    return rf, xgb

if __name__ == "__main__":
    df = load_all_data()
    print(f"Total rows: {len(df)}")
    rf_model, xgb_model = train_models(df)
