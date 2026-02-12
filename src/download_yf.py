from pathlib import Path
from datetime import datetime
import yfinance as yf
import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    "TSLA", "AMD", "AMZN", "NBIS", "NIO"
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

FUND_DIR = Path("data/raw/fundamentals")
FUND_DIR.mkdir(parents=True, exist_ok=True)
fund_rows = []

print("\nDownloading fundamentals...")

for ticker in TICKERS:
    print(f"Fundamentals for {ticker}...")
    tk = yf.Ticker(ticker)

    info = tk.get_info()

    fund_rows.append({
        "ticker": ticker,
        "date_pulled": datetime.today().strftime("%Y-%m-%d"),
        "trailing_eps": info.get("trailingEps"),
        "trailing_pe": info.get("trailingPE"),
        "total_revenue": info.get("totalRevenue"),
        "operating_cashflow": info.get("operatingCashflow"),
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector"),
    })

fund_df = pd.DataFrame(fund_rows)
fund_df.to_csv(FUND_DIR / "fundamentals.csv", index=False)

print("Fundamentals saved.")

print("Done.")
