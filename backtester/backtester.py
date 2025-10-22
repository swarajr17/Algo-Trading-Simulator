import pandas as pd

def backtest(data: pd.DataFrame, strategy_func, initial_capital=100000, **kwargs):
    """
    Backtests any strategy function following the template.
    """
    df = strategy_func(data, **kwargs).copy()
    df['Position'] = df['Signal'].shift(1).fillna(0)  # trade at next bar

    # Calculate returns
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Position']

    # Track equity curve
    df['Equity'] = (1 + df['Strategy_Return']).cumprod() * initial_capital

    results = {
        'Final_Capital': df['Equity'].iloc[-1],
        'Total_Return_%': (df['Equity'].iloc[-1] / initial_capital - 1) * 100,
        'Max_Drawdown_%': max_drawdown(df['Equity']),
        'Equity_Curve': df[['Equity']]
    }
    return results, df

def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return drawdown.min() * 100
