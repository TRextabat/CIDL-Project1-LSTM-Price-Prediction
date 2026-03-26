"""Load and normalize OHLCV data from FinAgent parquet files."""

from pathlib import Path

import pandas as pd


def load_asset(data_dir: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load and normalize OHLCV data from FinAgent parquet files.

    Handles differences between assets:
    - BTCUSD: column order (close,high,low,open,volume), tz-naive index named 'Date'
    - XAUUSD: standard column order, UTC-aware index named 'datetime'

    Returns:
        DataFrame with columns [open, high, low, close, volume],
        DatetimeIndex named 'datetime' in UTC, sorted ascending.
    """
    path = Path(data_dir) / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)

    # Normalize column names to lowercase
    col_map = {}
    for col in df.columns:
        col_map[col.lower()] = col
    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Ensure standard column order
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    df = df[required_cols]

    # Normalize index name
    if df.index.name != "datetime":
        df.index.name = "datetime"

    # Normalize timezone to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    # Validate no NaN in OHLC
    ohlc_nans = df[["open", "high", "low", "close"]].isna().sum()
    if ohlc_nans.any():
        raise ValueError(f"NaN values in OHLC for {symbol}_{timeframe}:\n{ohlc_nans[ohlc_nans > 0]}")

    # Fill missing volume with 0
    df["volume"] = df["volume"].fillna(0)

    return df.sort_index()
