import os
import pandas as pd
import yfinance as yf

# Path to store all cached data in ../data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_data(ticker, start="2015-01-01", end="2024-12-31", interval="1d"):
    """
    Fetch historical OHLCV data from Yahoo Finance for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. "AAPL", "MSFT").
    start : str
        Start date in format YYYY-MM-DD.
    end : str
        End date in format YYYY-MM-DD.
    interval : str
        Bar size ("1d", "1h", "15m", etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume.
    """
    # Suppress the FutureWarning by explicitly setting auto_adjust
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)

    # Check if valid data was returned
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[Error] No valid data returned for ticker: {ticker}")
        return pd.DataFrame()

    # Debug: Print the actual columns returned
    print(f"[Debug] Columns returned for {ticker}: {list(df.columns)}")
    print(f"[Debug] Data shape: {df.shape}")
    print(f"[Debug] Column index type: {type(df.columns)}")

    # Handle MultiIndex columns (common with single ticker downloads)
    if isinstance(df.columns, pd.MultiIndex):
        print("[Debug] Detected MultiIndex columns, flattening...")
        # For single ticker, take the first level (the actual column names)
        df.columns = [col[0] for col in df.columns]
        print(f"[Debug] Flattened columns: {list(df.columns)}")

    # Reset index so Date is a normal column instead of the index
    df.reset_index(inplace=True)

    # Drop rows with NaN values (market holidays, missing data, etc.)
    df.dropna(inplace=True)

    # Handle column name variations - yfinance might return different column names
    # Map common variations to standard names
    column_mapping = {
        'Adj Close': 'Adj Close',
        'Adj_Close': 'Adj Close',
        'AdjClose': 'Adj Close',
        'adj_close': 'Adj Close'
    }
    
    # Apply column mapping if needed
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    
    print(f"[Debug] Final columns after processing: {list(df.columns)}")
    
    # Validate schema (ensure all required columns exist)
    required_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[Error] Missing columns for {ticker}: {missing_cols}")
        print(f"[Error] Available columns: {list(df.columns)}")
        
        # Try to handle common missing column issues
        if "Adj Close" in missing_cols and "Close" in df.columns:
            print("[Fix] Creating 'Adj Close' from 'Close' column")
            df["Adj Close"] = df["Close"]
            missing_cols.remove("Adj Close")
        
        # If still missing critical columns, raise error
        if missing_cols:
            raise ValueError(f"Missing required columns in data for {ticker}: {missing_cols}")

    # Ensure proper data types
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Date to datetime if it's not already
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    print(f"[Debug] Final data shape: {df.shape}")
    print(f"[Debug] Sample of processed data:\n{df.head()}")

    return df


def save_data(df, ticker, start="2015-01-01", end="2024-12-31", interval="1d"):
    """
    Save DataFrame to CSV inside the /data folder.
    """
    filename = f"{ticker}_{interval}_{start}_{end}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    df.to_csv(filepath, index=False)
    print(f"[Info] Data saved to: {filepath}")
    return filepath


def load_data(ticker, start="2015-01-01", end="2024-12-31", interval="1d"):
    """
    Load data if already cached, otherwise fetch and save it.

    Returns
    -------
    pd.DataFrame
    """
    filename = f"{ticker}_{interval}_{start}_{end}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"[Info] Loading cached data from: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["Date"])
        return df
    else:
        print(f"[Info] Fetching new data for {ticker}")
        df = fetch_data(ticker, start, end, interval)
        if not df.empty:  # Save only if data is valid
            save_data(df, ticker, start, end, interval)
        return df