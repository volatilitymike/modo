import pandas as pd

def detect_entry_stage(intraday, lookback=9):
    """
    Stage 1 = F_numeric crosses TD Supply or Demand
      • Stage 1 Call = from ≤ TD Supply → > TD Supply
      • Stage 1 Put  = from ≥ TD Demand → < TD Demand
    Returns (stage, label, time) or (None, msg, None).
    """
    df = intraday.tail(lookback).copy()

    # Require all three columns
    required = ["F_numeric", "TD Supply Line F", "TD Demand Line F"]
    if any(col not in df.columns for col in required):
        return None, "Missing F_numeric or TD Supply/Demand Line F", None

    # Drop any rows where either line is NaN
    df = df.dropna(subset=["TD Supply Line F", "TD Demand Line F"])
    if len(df) < 2:
        return None, "Not enough data for Stage 1", None

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        prev_f, curr_f = prev["F_numeric"], curr["F_numeric"]
        prev_sup, curr_sup = prev["TD Supply Line F"], curr["TD Supply Line F"]
        prev_dem, curr_dem = prev["TD Demand Line F"], curr["TD Demand Line F"]
        ts = curr["Time"]

        # Stage 1 Call
        if prev_f <= prev_sup and curr_f > curr_sup:
            return 1, "Stage 1 Call: F% crossed above TD Supply", ts

        # Stage 1 Put
        if prev_f >= prev_dem and curr_f < curr_dem:
            return 1, "Stage 1 Put: F% crossed below TD Demand", ts

    return None, f"No Stage 1 call/put in last {lookback} bars", None


def detect_stage_2(intraday, lookback=9):
    """
    Stage 2 = F_numeric crosses Kijun_F.
      • Bullish cross (Call)  = from ≤ Kijun to > Kijun
      • Bearish cross (Put)   = from ≥ Kijun to < Kijun
    Returns (stage, label, time) or (None, msg, None).
    """
    df = intraday.tail(lookback).copy()
    if "F_numeric" not in df.columns or "Kijun_F" not in df.columns:
        return None, "Missing F_numeric or Kijun_F", None

    df = df[df["Kijun_F"].notna()]
    # need at least 2 rows to compare
    if len(df) < 2:
        return None, "Not enough data for Stage 2", None

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        prev_f, prev_k = prev["F_numeric"], prev["Kijun_F"]
        curr_f, curr_k = curr["F_numeric"], curr["Kijun_F"]
        ts = curr["Time"]

        # Bullish cross → Call
        if prev_f <= prev_k and curr_f > curr_k:
            return 2, "Stage 2 Call: F% crossed above Kijun", ts

        # Bearish cross → Put
        if prev_f >= prev_k and curr_f < curr_k:
            return 2, "Stage 2 Put: F% crossed below Kijun", ts

    return None, f"No Stage 2 call/put in last {lookback} bars", None
