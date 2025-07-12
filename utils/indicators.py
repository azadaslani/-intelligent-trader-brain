import pandas as pd


EPSILON = 1e-10

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI).

    This implementation avoids division by zero by adding a small epsilon value
    to the denominator when computing the relative strength (RS).
    """

    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / (loss + EPSILON)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow
