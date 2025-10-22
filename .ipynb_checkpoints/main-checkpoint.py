import argparse
from data_handler import load_data
from .strategy.moving_average import sma_crossover
from .backtester.backtester import backtest
from .backtester.metrics import calculate_cagr, calculate_sharpe, calculate_max_drawdown
from .utils.plotter import plot_equity_curve, plot_signals



def run_backtest(ticker, start, end, short_window, long_window, initial_capital):
    # 1. Load data
    df = load_data(ticker, start, end, "1d")

    if df.empty:
        print(f"[Error] No data for {ticker}.")
        return

    # 2. Run backtest
    results, df = backtest(
        df,
        sma_crossover,
        initial_capital=initial_capital,
        short_window=short_window,
        long_window=long_window,
    )

    # 3. Print metrics
    print("\n=== Backtest Results ===")
    print(f"Ticker: {ticker}")
    print(f"Period: {start} → {end}")
    print(f"Final Capital: ${results['Final_Capital']:.2f}")
    print(f"Total Return: {results['Total_Return_%']:.2f}%")
    print(f"CAGR: {calculate_cagr(df['Equity']):.2f}%")
    print(f"Sharpe Ratio: {calculate_sharpe(df['Strategy_Return']):.2f}")
    print(f"Max Drawdown: {results['Max_Drawdown_%']:.2f}%")

    # 4. Plots
    plot_equity_curve(df, ticker)
    plot_signals(df, ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algo Trading Simulator")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--short_window", type=int, default=50, help="Short SMA window")
    parser.add_argument("--long_window", type=int, default=200, help="Long SMA window")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")

    args = parser.parse_args()

    run_backtest(
        args.ticker, args.start, args.end, args.short_window, args.long_window, args.capital
    )
