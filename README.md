# 📈 Iran Stock Signal Detector

A machine learning system that detects buy signals for Tehran Stock Exchange (TSE) stocks using technical indicators and price action features.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-XGBoost | RandomForest-orange)

---

## 🎯 What it does

- Downloads real historical data from TSE (TSETMC)
- Engineers 15+ technical features (RSI, MACD, Bollinger Bands, momentum, volume ratio...)
- Trains XGBoost and Random Forest classifiers to predict if a stock will rise 5%+ in the next 5 days
- Displays buy probabilities and charts in an interactive dashboard

---

## 📊 Dashboard Preview

- **Latest Signals table** — all stocks ranked by buy probability
- **Price Chart** — historical close price per stock
- **RSI Chart** — with overbought/oversold levels

---

## 🗂️ Project Structure

```
├── data/
│   ├── raw/          # Raw OHLCV data from TSE
│   └── processed/    # Feature-engineered data
├── src/
│   ├── scraper.py    # Downloads data via pytse-client
│   ├── features.py   # Technical indicator engineering
│   └── model.py      # Model training and evaluation
├── dashboard/
│   └── app.py        # Streamlit dashboard
└── requirements.txt
```

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/iran-stock-signal-detector.git
cd iran-stock-signal-detector

pip install -r requirements.txt
```

---

## 🚀 Run

```bash
# Step 1 - Download data
python src/scraper.py

# Step 2 - Engineer features
python src/features.py

# Step 3 - Train and evaluate models
python src/model.py

# Step 4 - Launch dashboard
streamlit run dashboard/app.py
```

---

## 🔬 Features Used

| Feature | Description |
|---|---|
| RSI | Relative Strength Index (14) |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands | Upper and lower bands |
| SMA 20/50 | Simple moving averages |
| Volume Ratio | Today's volume vs 20-day average |
| Momentum 5/10 | Price change over 5 and 10 days |
| 52W High/Low Distance | Distance from yearly high and low |
| Price to SMA | Close price relative to SMA |

---

## 📉 Model Performance

| Model | Precision | Notes |
|---|---|---|
| Random Forest | 0.28 | class_weight=balanced |
| XGBoost | 0.25 | scale_pos_weight=4 |

> ⚠️ Low precision is expected due to class imbalance (buy signals are rare by nature). This project is for educational purposes only — not financial advice.

---

## 📦 Data Source

Data is fetched via [pytse-client](https://github.com/Glyphack/pytse-client) from TSETMC (Tehran Securities Exchange Technology Management Co.)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. Do not use it for actual trading decisions.
