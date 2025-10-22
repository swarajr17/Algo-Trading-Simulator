# utils/data_loader.py
from data_handler.data_handler import load_data

def get_data(ticker: str, start="2015-01-01", end="2024-12-31", interval="1d"):
    """
    Wrapper around data_handler.load_data to simplify imports.
    """
    return load_data(ticker, start, end, interval)
