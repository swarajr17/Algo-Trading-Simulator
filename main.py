import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from your modules
from data_handler.data_handler import load_data
from strategies.moving_average import sma_crossover
from backtester.backtester import backtest, max_drawdown
from utils.plotter import plot_equity_curve, plot_signals


def calculate_cagr(equity_series):
    """
    Calculate Compound Annual Growth Rate
    """
    import pandas as pd
    
    if len(equity_series) < 2:
        return 0
    
    initial_value = equity_series.iloc[0]
    final_value = equity_series.iloc[-1]
    
    # Calculate years (assuming daily data)
    n_days = len(equity_series)
    n_years = n_days / 252  # Trading days per year
    
    if n_years == 0 or initial_value == 0:
        return 0
    
    cagr = ((final_value / initial_value) ** (1 / n_years) - 1) * 100
    return cagr


def calculate_sharpe(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio
    Assumes returns are daily returns
    """
    import numpy as np
    
    # Clean returns (remove NaN and inf)
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0
    
    # Annual Sharpe ratio (assuming 252 trading days)
    mean_return = returns_clean.mean() * 252
    std_return = returns_clean.std() * np.sqrt(252)
    
    if std_return == 0:
        return 0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe


def run_backtest(ticker, start, end, short_window, long_window, initial_capital):
    # 1. Load data
    df = load_data(ticker, start, end, "1d")

    if df.empty:
        print(f"[Error] No data for {ticker}.")
        return

    # 2. Run backtest
    results, df_with_signals = backtest(
        df,
        sma_crossover,
        initial_capital=initial_capital,
        short_window=short_window,
        long_window=long_window,
    )

    # 3. Print metrics
    print("\n=== Backtest Results ===")
    print(f"Ticker: {ticker}")
    print(f"Period: {start} -> {end}")  # Fixed: Changed → to ->
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${results['Final_Capital']:,.2f}")
    print(f"Total Return: {results['Total_Return_%']:.2f}%")
    
    # Calculate additional metrics
    cagr = calculate_cagr(df_with_signals['Equity'])
    sharpe = calculate_sharpe(df_with_signals['Strategy_Return'])
    
    print(f"CAGR: {cagr:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {results['Max_Drawdown_%']:.2f}%")

    # 4. Plots
    plot_equity_curve(df_with_signals, ticker)
    plot_signals(df_with_signals, ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algo Trading Simulator")
    parser.add_argument("--ticker", type=str, default="MSFT", help="Stock ticker")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--short_window", type=int, default=50, help="Short SMA window")
    parser.add_argument("--long_window", type=int, default=200, help="Long SMA window")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")

    args = parser.parse_args()

    run_backtest(
        args.ticker, args.start, args.end, args.short_window, args.long_window, args.capital
    )