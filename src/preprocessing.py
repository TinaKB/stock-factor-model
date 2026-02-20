from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PRICE_REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PRICE_DROP_COLS_IF_PRESENT = ["Dividends", "Stock Splits"]


def clean_price_df(price: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans one ticker's price dataframe:
    - validate required columns
    - parse Date with utc=True (prevents mixed timezone errors)
    - remove timezone info (daily data)
    - sort, set index
    - drop optional columns if present
    - keep core OHLCV columns
    - convert numeric
    - forward-fill missing values, drop remaining NA
    - create return + log_return
    - clip extreme returns
    """
    # validate columns
    missing = [c for c in PRICE_REQUIRED_COLS if c not in price.columns]
    if missing:
        raise ValueError(f"Price CSV missing required columns: {missing}")

    # --- parse Date with forced UTC (prevents mixed timezone errors) ---
    price = price.copy()
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce", utc=True)

    if price["Date"].isna().any():
        bad_rows = price.loc[price["Date"].isna()].head(5)
        raise ValueError(
            "Some Date values could not be parsed. Example rows:\n"
            f"{bad_rows}"
        )

    # remove timezone info (we only need daily resolution)
    price["Date"] = price["Date"].dt.tz_convert(None)

    # sort and set index
    price = price.sort_values("Date").set_index("Date")

    # drop optional columns if present
    for c in PRICE_DROP_COLS_IF_PRESENT:
        if c in price.columns:
            price = price.drop(columns=[c])

    # keep only core OHLCV columns (if extra columns exist)
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in price.columns]
    price = price[keep_cols].copy()

    # convert numeric
    for c in keep_cols:
        price[c] = pd.to_numeric(price[c], errors="coerce")

    # missing handling
    price = price.ffill()
    price = price.dropna()

    # create returns
    price["return"] = price["Close"].pct_change()
    price["log_return"] = np.log(price["Close"]).diff()

    # remove first row NA due to diff
    price = price.dropna()

    # clip extreme daily returns (finance-friendly outlier handling)
    price["return"] = price["return"].clip(-0.30, 0.30)
    price["log_return"] = price["log_return"].clip(-0.30, 0.30)

    return price

def clean_fundamentals_df(fund: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans fundamentals snapshot:
    - parse date_pulled
    - ensure numeric columns numeric
    - fill missing numeric values with column median
    - one-hot encode sector
    """
    required = ["ticker", "date_pulled", "sector"]
    missing = [c for c in required if c not in fund.columns]
    if missing:
        raise ValueError(f"Fundamentals CSV missing required columns: {missing}")

    fund = fund.copy()
    fund["date_pulled"] = pd.to_datetime(fund["date_pulled"], errors="coerce")

    # convert likely numeric columns (everything except ticker/date/sector)
    non_num = {"ticker", "date_pulled", "sector"}
    numeric_cols = [c for c in fund.columns if c not in non_num]
    for c in numeric_cols:
        fund[c] = pd.to_numeric(fund[c], errors="coerce")

    # fill numeric NA with median
    med = fund[numeric_cols].median(numeric_only=True)
    fund[numeric_cols] = fund[numeric_cols].fillna(med)

    # one-hot encode sector
    fund = pd.get_dummies(fund, columns=["sector"], drop_first=True)

    return fund


def attach_fundamentals_to_prices(
    price: pd.DataFrame,
    fund: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Attaches fundamentals (snapshot) to each row in a ticker's price df.
    """
    row = fund[fund["ticker"] == ticker]
    if row.empty:
        raise ValueError(f"No fundamentals found for ticker={ticker}")

    # take the first matching row
    row = row.iloc[0].to_dict()

    out = price.copy()
    for k, v in row.items():
        if k in ("ticker", "date_pulled"):
            continue
        out[k] = v

    return out


def standardize_numeric_features(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numeric columns in df, returning scaled df + fitted scaler.
    NOTE: For real ML, fit scaler on train split only.
    """
    exclude_cols = exclude_cols or []
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in exclude_cols]

    scaler = StandardScaler()
    scaled = df.copy()
    scaled[num_cols] = scaler.fit_transform(scaled[num_cols])

    return scaled, scaler


def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Returns a structured data quality report:
    - row/column counts
    - top missing %
    - summary stats for key columns
    - preprocessing fixes applied
    """

    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_pct_top10": missing_pct.head(10).to_dict(),
        "fixes_applied": [
            "Parsed Date column with utc=True",
            "Removed timezone information",
            "Sorted by Date and set as index",
            "Dropped Dividends and Stock Splits (if present)",
            "Kept core OHLCV columns",
            "Converted numeric columns using pd.to_numeric",
            "Forward-filled missing values",
            "Dropped remaining NA rows",
            "Created return and log_return features",
            "Clipped extreme returns to ±30%"
        ]
    }

    # Add summary statistics for important columns if they exist
    for col in ["Close", "Volume", "return", "log_return"]:
        if col in df.columns:
            s = df[col]
            report[f"{col}_summary"] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }

    return report