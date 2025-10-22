import pandas as pd

def sma_crossover(data: pd.DataFrame, short_window=50, long_window=200) -> pd.DataFrame:
    """
    Moving Average Crossover Strategy
    - Buy signal (1) when short SMA > long SMA
    - Sell signal (-1) when short SMA < long SMA
    - Hold (0) otherwise

    Returns:
        DataFrame with 'Signal' column added.
    """
    df = data.copy()

    # Compute moving averages (require full window before producing a value)
    df['SMA_short'] = df['Adj Close'].rolling(window=short_window).mean()
    df['SMA_long']  = df['Adj Close'].rolling(window=long_window).mean()

    # Initialize signals
    df['Signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1   # Buy
    df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = -1  # Sell

    return df
