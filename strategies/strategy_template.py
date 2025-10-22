import pandas as pd

def strategy_template(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Every strategy must:
    - Take in a DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'.
    - Return the same DataFrame with a new column 'Signal':
      1 for buy/long, -1 for sell/short, 0 for no trade.
    """
    df = data.copy()
    df['Signal'] = 0  # default no trade
    # --- Your logic here ---
    
    return df
