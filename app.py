import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from data_handler.data_handler import load_data
from strategies.moving_average import sma_crossover
from backtester.backtester import backtest


def calculate_cagr(equity_series):
    """Calculate Compound Annual Growth Rate"""
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
    """Calculate Sharpe Ratio"""
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


def calculate_additional_metrics(df_with_signals):
    """Calculate additional performance metrics"""
    returns = df_with_signals['Strategy_Return'].dropna()
    equity = df_with_signals['Equity']
    
    metrics = {}
    
    # Win rate
    winning_trades = (returns > 0).sum()
    total_trades = len(returns[returns != 0])
    metrics['Win Rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    metrics['Avg Win'] = wins.mean() * 100 if len(wins) > 0 else 0
    metrics['Avg Loss'] = losses.mean() * 100 if len(losses) > 0 else 0
    
    # Profit factor
    total_wins = wins.sum()
    total_losses = abs(losses.sum())
    metrics['Profit Factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Volatility (annualized)
    metrics['Volatility'] = returns.std() * np.sqrt(252) * 100
    
    return metrics


def create_equity_curve_plot(df_with_signals, ticker):
    """Create interactive equity curve plot"""
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=df_with_signals['Date'],
        y=df_with_signals['Equity'],
        mode='lines',
        name='Strategy Equity',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Buy and Hold comparison
    initial_capital = df_with_signals['Equity'].iloc[0]
    buy_hold = (df_with_signals['Close'] / df_with_signals['Close'].iloc[0]) * initial_capital
    
    fig.add_trace(go.Scatter(
        x=df_with_signals['Date'],
        y=buy_hold,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Equity Curve - {ticker}',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig


def create_price_signals_plot(df_with_signals, ticker, short_window, long_window):
    """Create price chart with signals"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'{ticker} Price & Signals', 'Volume'],
        row_width=[0.7, 0.3]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(
        x=df_with_signals['Date'],
        y=df_with_signals['Close'],
        mode='lines',
        name='Price',
        line=dict(color='black', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_with_signals['Date'],
        y=df_with_signals[f'SMA_{short_window}'],
        mode='lines',
        name=f'SMA {short_window}',
        line=dict(color='blue', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_with_signals['Date'],
        y=df_with_signals[f'SMA_{long_window}'],
        mode='lines',
        name=f'SMA {long_window}',
        line=dict(color='red', width=1)
    ), row=1, col=1)
    
    # Buy signals - check for Position column or signal columns
    buy_condition = None
    if 'Position' in df_with_signals.columns:
        buy_signals = df_with_signals[df_with_signals['Position'] == 1]
        buy_condition = 'Position'
    elif 'Signal' in df_with_signals.columns:
        buy_signals = df_with_signals[df_with_signals['Signal'] == 1]
        buy_condition = 'Signal'
    elif 'Buy_Signal' in df_with_signals.columns:
        buy_signals = df_with_signals[df_with_signals['Buy_Signal'] == 1]
        buy_condition = 'Buy_Signal'
    else:
        buy_signals = pd.DataFrame()  # Empty dataframe if no signal column found
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['Date'],
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ), row=1, col=1)
    
    # Sell signals
    sell_condition = None
    if 'Position' in df_with_signals.columns:
        sell_signals = df_with_signals[df_with_signals['Position'] == -1]
        sell_condition = 'Position'
    elif 'Signal' in df_with_signals.columns:
        sell_signals = df_with_signals[df_with_signals['Signal'] == -1]
        sell_condition = 'Signal'
    elif 'Sell_Signal' in df_with_signals.columns:
        sell_signals = df_with_signals[df_with_signals['Sell_Signal'] == 1]
        sell_condition = 'Sell_Signal'
    else:
        sell_signals = pd.DataFrame()  # Empty dataframe if no signal column found
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['Date'],
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=1, col=1)
    
    print(f"[Debug] Buy condition column: {buy_condition}, Sell condition column: {sell_condition}")
    print(f"[Debug] Buy signals found: {len(buy_signals)}, Sell signals found: {len(sell_signals)}")
    
    # Volume
    fig.add_trace(go.Bar(
        x=df_with_signals['Date'],
        y=df_with_signals['Volume'],
        name='Volume',
        marker_color='lightblue',
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        title=f'{ticker} - Price Action & Trading Signals',
        xaxis_title='Date',
        height=700,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def create_returns_distribution_plot(df_with_signals):
    """Create returns distribution plot"""
    returns = df_with_signals['Strategy_Return'].dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Strategy Returns Distribution',
        xaxis_title='Daily Returns (%)',
        yaxis_title='Frequency',
        height=400
    )
    
    return fig


def run_backtest_ui(ticker, start_date, end_date, short_window, long_window, initial_capital):
    """Run backtest and return results"""
    try:
        # Load data
        with st.spinner(f'Loading data for {ticker}...'):
            df = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), "1d")
        
        if df.empty:
            st.error(f"No data available for {ticker} in the specified date range.")
            return None, None
        
        # Run backtest
        with st.spinner('Running backtest...'):
            results, df_with_signals = backtest(
                df,
                sma_crossover,
                initial_capital=initial_capital,
                short_window=short_window,
                long_window=long_window,
            )
        
        return results, df_with_signals
    
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        return None, None


def main():
    st.set_page_config(
        page_title="Algo Trading Simulator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Algorithmic Trading Simulator")
    st.markdown("---")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("🔧 Backtest Parameters")
        
        # Stock selection
        ticker = st.text_input("Stock Ticker", value="MSFT", help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)")
        
        # Date range
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2015, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=date(2024, 12, 31))
        
        # Strategy parameters
        st.subheader("📊 Strategy Parameters")
        short_window = st.slider("Short SMA Window", min_value=5, max_value=100, value=50, step=5)
        long_window = st.slider("Long SMA Window", min_value=50, max_value=300, value=200, step=10)
        
        if short_window >= long_window:
            st.error("Short window must be less than long window!")
        
        # Capital
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)
        
        # Run backtest button
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Main content area
    if run_button and short_window < long_window:
        results, df_with_signals = run_backtest_ui(ticker, start_date, end_date, short_window, long_window, initial_capital)
        
        if results is not None and df_with_signals is not None:
            # Calculate additional metrics
            cagr = calculate_cagr(df_with_signals['Equity'])
            sharpe = calculate_sharpe(df_with_signals['Strategy_Return'])
            additional_metrics = calculate_additional_metrics(df_with_signals)
            
            # Display key metrics
            st.header("📊 Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{results['Total_Return_%']:.2f}%",
                    delta=f"{results['Total_Return_%']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "CAGR",
                    f"{cagr:.2f}%",
                    delta=f"{cagr:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    delta=f"{sharpe:.2f}"
                )
            
            with col4:
                st.metric(
                    "Max Drawdown",
                    f"{results['Max_Drawdown_%']:.2f}%",
                    delta=f"-{results['Max_Drawdown_%']:.2f}%"
                )
            
            # Detailed metrics table
            st.subheader("📈 Detailed Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_data = {
                    "Metric": [
                        "Initial Capital",
                        "Final Capital",
                        "Total Return",
                        "CAGR",
                        "Sharpe Ratio",
                        "Max Drawdown"
                    ],
                    "Value": [
                        f"${initial_capital:,.2f}",
                        f"${results['Final_Capital']:,.2f}",
                        f"{results['Total_Return_%']:.2f}%",
                        f"{cagr:.2f}%",
                        f"{sharpe:.2f}",
                        f"{results['Max_Drawdown_%']:.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
            
            with col2:
                additional_data = {
                    "Metric": [
                        "Win Rate",
                        "Average Win",
                        "Average Loss",
                        "Profit Factor",
                        "Volatility (Annual)"
                    ],
                    "Value": [
                        f"{additional_metrics['Win Rate']:.2f}%",
                        f"{additional_metrics['Avg Win']:.2f}%",
                        f"{additional_metrics['Avg Loss']:.2f}%",
                        f"{additional_metrics['Profit Factor']:.2f}",
                        f"{additional_metrics['Volatility']:.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(additional_data), use_container_width=True)
            
            # Charts
            st.header("📊 Charts")
            
            # Equity curve
            st.subheader("Equity Curve")
            equity_fig = create_equity_curve_plot(df_with_signals, ticker)
            st.plotly_chart(equity_fig, use_container_width=True)
            
            # Price and signals
            st.subheader("Price Chart with Signals")
            signals_fig = create_price_signals_plot(df_with_signals, ticker, short_window, long_window)
            st.plotly_chart(signals_fig, use_container_width=True)
            
            # Returns distribution
            st.subheader("Returns Distribution")
            returns_fig = create_returns_distribution_plot(df_with_signals)
            st.plotly_chart(returns_fig, use_container_width=True)
            
            # Data table
            with st.expander("📋 View Raw Data", expanded=False):
                st.dataframe(df_with_signals.tail(100), use_container_width=True)
    
    elif not run_button:
        # Instructions
        st.info("""
        ### Welcome to the Algorithmic Trading Simulator! 
        
        This tool allows you to backtest a Simple Moving Average (SMA) crossover strategy on any stock.
        
        **How to use:**
        1. 📝 Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
        2. 📅 Select your backtest date range
        3. ⚙️ Adjust the SMA windows (short must be < long)
        4. 💰 Set your initial capital
        5. 🚀 Click "Run Backtest" to see results
        
        **Strategy Explanation:**
        - **Buy Signal**: When short SMA crosses above long SMA
        - **Sell Signal**: When short SMA crosses below long SMA
        - The strategy alternates between being long the stock and holding cash
        
        Select your parameters in the sidebar and click "Run Backtest" to get started!
        """)


if __name__ == "__main__":
    main()