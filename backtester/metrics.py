# backtester/metrics.py

import numpy as np
import pandas as pd


def calculate_cagr(equity_series):
    """
    Calculate Compound Annual Growth Rate from an equity curve.
    
    Parameters
    ----------
    equity_series : pd.Series
        Series of portfolio values over time
    
    Returns
    -------
    float
        CAGR as a percentage
    """
    if len(equity_series) < 2:
        return 0
    
    initial_value = equity_series.iloc[0]
    final_value = equity_series.iloc[-1]
    
    # Calculate years (assuming daily data with ~252 trading days per year)
    n_days = len(equity_series)
    n_years = n_days / 252
    
    if n_years == 0 or initial_value == 0:
        return 0
    
    cagr = ((final_value / initial_value) ** (1 / n_years) - 1) * 100
    return cagr


def calculate_sharpe(returns, risk_free_rate=0.02):
    """
    Calculate annualized Sharpe Ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    risk_free_rate : float
        Annual risk-free rate (default 2%)
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    # Clean returns (remove NaN and inf)
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0
    
    # Annualize metrics (assuming 252 trading days)
    mean_return = returns_clean.mean() * 252
    std_return = returns_clean.std() * np.sqrt(252)
    
    if std_return == 0:
        return 0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe


def calculate_max_drawdown(equity_series):
    """
    Calculate maximum drawdown from an equity curve.
    
    Parameters
    ----------
    equity_series : pd.Series
        Series of portfolio values over time
    
    Returns
    -------
    float
        Maximum drawdown as a percentage
    """
    if len(equity_series) < 2:
        return 0
    
    # Calculate running maximum
    running_max = equity_series.cummax()
    
    # Calculate drawdown
    drawdown = (equity_series - running_max) / running_max
    
    # Return max drawdown as percentage
    return drawdown.min() * 100


def calculate_sortino(returns, risk_free_rate=0.02, target_return=0):
    """
    Calculate Sortino Ratio (like Sharpe but only considers downside volatility).
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    risk_free_rate : float
        Annual risk-free rate
    target_return : float
        Target return for downside deviation calculation
    
    Returns
    -------
    float
        Annualized Sortino ratio
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0
    
    # Calculate downside returns (only negative returns)
    downside_returns = returns_clean[returns_clean < target_return]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside risk
    
    # Annualize metrics
    mean_return = returns_clean.mean() * 252
    downside_std = downside_returns.std() * np.sqrt(252)
    
    if downside_std == 0:
        return np.inf
    
    sortino = (mean_return - risk_free_rate) / downside_std
    return sortino


def calculate_calmar(equity_series):
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).
    
    Parameters
    ----------
    equity_series : pd.Series
        Series of portfolio values over time
    
    Returns
    -------
    float
        Calmar ratio
    """
    cagr = calculate_cagr(equity_series)
    max_dd = abs(calculate_max_drawdown(equity_series))
    
    if max_dd == 0:
        return np.inf
    
    return cagr / max_dd


def calculate_win_rate(returns):
    """
    Calculate the percentage of winning trades.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    
    Returns
    -------
    float
        Win rate as a percentage
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0
    
    winning_trades = (returns_clean > 0).sum()
    total_trades = len(returns_clean[returns_clean != 0])
    
    if total_trades == 0:
        return 0
    
    return (winning_trades / total_trades) * 100