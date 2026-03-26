"""Compute 20 technical features from OHLCV data using pandas-ta."""

import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20 technical features from OHLCV DataFrame.

    Features computed:
        1.  log_returns          - log(close / close.shift(1))
        2.  abs_return           - abs(log_returns)
        3.  high_low_range       - (high - low) / close
        4.  close_open_range     - (close - open) / close
        5.  rsi_14              - RSI(14)
        6.  macd_hist           - MACD histogram (12, 26, 9)
        7.  stoch_k             - Stochastic %K (14, 3, 3)
        8.  stoch_d             - Stochastic %D (14, 3, 3)
        9.  atr_norm            - ATR(14) / close
        10. bb_width            - (BBU - BBL) / BBM
        11. bb_position         - (close - BBL) / (BBU - BBL)
        12. ema_20_ratio        - close / EMA(20) - 1
        13. ema_50_ratio        - close / EMA(50) - 1
        14. sma_50_ratio        - close / SMA(50) - 1
        15. roc_10              - Rate of Change(10)
        16. volume_ratio        - volume / volume.rolling(20).mean()
        17. hour_sin            - sin(2*pi*hour/24)  [intraday only]
        18. hour_cos            - cos(2*pi*hour/24)  [intraday only]
        19. day_sin             - sin(2*pi*dayofweek/5)
        20. day_cos             - cos(2*pi*dayofweek/5)

    Args:
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        DataFrame with 20 feature columns, NaN rows dropped.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # 1-4: Price-based features
    feat["log_returns"] = np.log(close / close.shift(1))
    feat["abs_return"] = feat["log_returns"].abs()
    feat["high_low_range"] = (high - low) / close
    feat["close_open_range"] = (close - open_) / close

    # 5: RSI
    rsi = ta.rsi(close, length=14)
    feat["rsi_14"] = rsi / 100.0  # Normalize to [0, 1]

    # 6: MACD histogram
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    feat["macd_hist"] = macd_df["MACDh_12_26_9"]

    # 7-8: Stochastic
    stoch_df = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    feat["stoch_k"] = stoch_df["STOCHk_14_3_3"] / 100.0
    feat["stoch_d"] = stoch_df["STOCHd_14_3_3"] / 100.0

    # 9: ATR normalized
    atr = ta.atr(high, low, close, length=14)
    feat["atr_norm"] = atr / close

    # 10-11: Bollinger Bands
    bb_df = ta.bbands(close, length=20, std=2)
    # pandas-ta column names vary by version; find dynamically
    bbu_col = [c for c in bb_df.columns if c.startswith("BBU")][0]
    bbl_col = [c for c in bb_df.columns if c.startswith("BBL")][0]
    bbm_col = [c for c in bb_df.columns if c.startswith("BBM")][0]
    bbu = bb_df[bbu_col]
    bbl = bb_df[bbl_col]
    bbm = bb_df[bbm_col]
    feat["bb_width"] = (bbu - bbl) / bbm
    bb_range = bbu - bbl
    feat["bb_position"] = np.where(
        bb_range > 0,
        (close - bbl) / bb_range,
        0.5  # Default when bands are flat
    )

    # 12-14: Moving average ratios
    ema_20 = ta.ema(close, length=20)
    ema_50 = ta.ema(close, length=50)
    sma_50 = ta.sma(close, length=50)
    feat["ema_20_ratio"] = close / ema_20 - 1
    feat["ema_50_ratio"] = close / ema_50 - 1
    feat["sma_50_ratio"] = close / sma_50 - 1

    # 15: Rate of change
    feat["roc_10"] = ta.roc(close, length=10) / 100.0

    # 16: Volume ratio
    vol_mean = volume.rolling(20).mean()
    if (volume == 0).all():
        feat["volume_ratio"] = 0.0
    else:
        feat["volume_ratio"] = np.where(
            vol_mean > 0,
            volume / vol_mean,
            0.0
        )

    # 17-18: Hour encoding (only for intraday data)
    hours = df.index.hour
    is_intraday = hours.nunique() > 1
    if is_intraday:
        feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    else:
        feat["hour_sin"] = 0.0
        feat["hour_cos"] = 0.0

    # 19-20: Day-of-week encoding
    dow = df.index.dayofweek
    feat["day_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["day_cos"] = np.cos(2 * np.pi * dow / 5)

    # Drop rows with NaN (from indicator warmup, typically first ~50 rows)
    feat = feat.dropna()

    return feat
