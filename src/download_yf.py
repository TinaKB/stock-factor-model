from pathlib import Path
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    "TSLA", "AMD", "AMZN", "MSTR", "BTC-USD"
]

DATA_DIR = Path("data/raw/prices")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.Ticker(ticker).history(period="5y")

    if df.empty:
        print(f"No data for {ticker}")
        continue

    df.to_csv(DATA_DIR / f"{ticker}.csv")

print("Done.")
