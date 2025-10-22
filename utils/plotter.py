# utils/plotter.py
import matplotlib.pyplot as plt

def plot_equity_curve(df, ticker):
    """
    Plots the strategy equity curve.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Equity'], label="Strategy Equity")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(f"Equity Curve - {ticker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_signals(df, ticker):
    """
    Plot price with buy/sell markers from 'Signal' column.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Adj Close'], label="Price", alpha=0.6)

    buys = df[df['Signal'] == 1]
    sells = df[df['Signal'] == -1]

    plt.scatter(buys['Date'], buys['Adj Close'], marker="^", color="green", label="Buy", alpha=0.8)
    plt.scatter(sells['Date'], sells['Adj Close'], marker="v", color="red", label="Sell", alpha=0.8)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Price & Signals - {ticker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
