from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from preprocessing import (
    clean_price_df,
    clean_fundamentals_df,
    attach_fundamentals_to_prices,
    standardize_numeric_features,
    data_quality_report,
)
from utils import ensure_dir, read_csv_safely, to_parquet


# =========================
# CONFIG (matches your folders)
# =========================
RAW_PRICES_DIR = Path("data/raw/prices")  # ABNB.csv, AAPL.csv, ...
RAW_FUNDAMENTALS_PATH = Path("data/raw/fundamentals/fundamentals.csv")

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("data/processed/reports")


def infer_ticker_from_filename(path: Path) -> str:
    """
    Expects filenames like 'ABNB.csv' -> returns 'ABNB'
    """
    return path.stem.upper()


def main(scale: bool = False) -> None:
    # 1) Ensure output folders exist
    ensure_dir(PROCESSED_DIR)
    ensure_dir(REPORTS_DIR)

    # 2) Load + clean fundamentals (one row per ticker)
    fund_raw = read_csv_safely(RAW_FUNDAMENTALS_PATH)
    fund = clean_fundamentals_df(fund_raw)

    # 3) Find all price CSV files automatically (no need to know names)
    price_files = sorted(RAW_PRICES_DIR.glob("*.csv"))
    if not price_files:
        raise FileNotFoundError(
            f"No price CSVs found in {RAW_PRICES_DIR.resolve()}.\n"
            f"Expected files like ABNB.csv, AAPL.csv, TSLA.csv, etc."
        )

    print("Found price files:")
    for f in price_files:
        print("  -", f.name)

    # 4) Process each ticker
    for pf in price_files:
        ticker = infer_ticker_from_filename(pf)
        print(f"\n=== Processing {ticker} ===")

        # 4a) Read raw price CSV
        price_raw = read_csv_safely(pf)

        # 4b) Clean price data (Date parsing, index, returns, clip outliers, etc.)
        price_clean = clean_price_df(price_raw)

        # 4c) Attach fundamentals snapshot features for this ticker
        merged = attach_fundamentals_to_prices(price_clean, fund, ticker)

        # 4d) Optional scaling (not required for Week 2)
        if scale:
            merged, _ = standardize_numeric_features(merged)

        # 4e) Save cleaned dataset (Parquet)
        out_parquet = PROCESSED_DIR / f"{ticker}_cleaned.parquet"
        to_parquet(merged, out_parquet)
        print(f"Saved parquet: {out_parquet}")

        # 4f) Create report dict
        report = data_quality_report(merged)

        # 4g) Save report JSON (structured)
        report_json_path = REPORTS_DIR / f"{ticker}_report.json"
        report_json_path.write_text(json.dumps(report, indent=2))
        print(f"Saved JSON report: {report_json_path}")

        # 4h) Save report CSV (flattened one-row)
        report_df = pd.json_normalize(report)
        report_csv_path = REPORTS_DIR / f"{ticker}_report.csv"
        report_df.to_csv(report_csv_path, index=False)
        print(f"Saved CSV report: {report_csv_path}")

    print("\n✅ Done.")
    print(f"Processed datasets saved to: {PROCESSED_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    # Keep scale=False for Week 2 (scaling is usually done after train/test split)
    main(scale=False)