import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')
# Dashboard Configuration
st.set_page_config(page_title="Portfolio Performance Dashboard", layout="wide")

# Title and Description
st.title("Portfolio Performance Tracker")
st.markdown("**Track the optimized portfolio performance from buy date to current**")

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Default tickers and weights (your optimized portfolio)
default_tickers = ["HWM", "NVDA", "MSI", "AMZN", "MA", "TSLA", "ALB"]
default_weights = [0, 0, 15.6, 12.5, 28.9, 33.8, 9.2]  

# User inputs
buy_date = st.sidebar.date_input(
    "Portfolio Buy Date",
    value=date(2025, 9, 1),
    min_value=date(2016, 11, 1),
    max_value=date.today()
)

initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000,
    value=10000,
    step=1000
)

# Portfolio weights input
st.sidebar.subheader("Portfolio Weights (%)")
weights = {}
for i, ticker in enumerate(default_tickers):
    weights[ticker] = st.sidebar.number_input(
        f"{ticker}",
        min_value=0.0,
        max_value=100.0,
        value=float(default_weights[i]),
        step=0.1,
        key=f"weight_{ticker}"
    )

# Normalize weights to sum to 100%
total_weight = sum(weights.values())
if total_weight != 100:
    st.sidebar.warning(f"Weights sum to {total_weight:.1f}%. Normalizing to 100%.")
    weights = {k: (v/total_weight)*100 for k, v in weights.items()}

# Main dashboard
if st.sidebar.button("Update Portfolio Performance", type="primary"):
    
    try:
        # Download data with better error handling
        with st.spinner("Fetching market data..."):
            # Filter out tickers with 0 weight
            active_tickers = [t for t in default_tickers if weights[t] > 0]
            
            if not active_tickers:
                st.error("Please assign weights to at least one ticker.")
                st.stop()
            
            # Method 1: Try downloading all tickers at once
            try:
                end_date = date.today()
                # Download all active tickers together
                data_download = yf.download(
                    active_tickers, 
                    start=buy_date, 
                    end=end_date,
                    progress=False,
                    group_by='ticker',
                    auto_adjust=True,
                    threads=False
                )
                
                # Handle single ticker vs multiple tickers
                if len(active_tickers) == 1:
                    # Single ticker - data structure is different
                    if 'Close' in data_download.columns:
                        data = pd.DataFrame({active_tickers[0]: data_download['Close']})
                    else:
                        data = pd.DataFrame({active_tickers[0]: data_download})
                else:
                    # Multiple tickers - extract Close prices
                    close_data = {}
                    for ticker in active_tickers:
                        if ticker in data_download.columns.levels[0]:
                            close_data[ticker] = data_download[ticker]['Close']
                        elif 'Close' in data_download.columns:
                            # Fallback for different data structure
                            close_data[ticker] = data_download['Close']
                    
                    if close_data:
                        data = pd.DataFrame(close_data)
                    else:
                        # Try alternative structure
                        data = pd.DataFrame()
                        for ticker in active_tickers:
                            try:
                                data[ticker] = data_download[ticker]['Close']
                            except:
                                pass
                
            except Exception as e:
                st.info("Trying alternative download method...")
                # Method 2: Download each ticker individually
                data_dict = {}
                failed_tickers = []
                
                for ticker in active_tickers:
                    try:
                        ticker_data = yf.Ticker(ticker)
                        end_date = date.today()
                        hist = ticker_data.history(start=buy_date, end=end_date, auto_adjust=True)
                        
                        if not hist.empty and 'Close' in hist.columns:
                            data_dict[ticker] = hist['Close']
                        else:
                            failed_tickers.append(ticker)
                    except Exception as ticker_error:
                        failed_tickers.append(ticker)
                        st.warning(f"Could not fetch data for {ticker}: {str(ticker_error)}")
                
                if failed_tickers and len(failed_tickers) == len(active_tickers):
                    st.error("Could not fetch data for any ticker. Please check your internet connection.")
                    st.stop()
                
                if data_dict:
                    # Create DataFrame from the successfully downloaded data
                    data = pd.DataFrame(data_dict)
                else:
                    st.error("No data could be downloaded.")
                    st.stop()
            
            # Clean the data
            data = data.ffill().bfill()  # Forward fill then backward fill
            data = data.dropna()  # Remove any remaining NaN values
            
            if data.empty:
                st.error("No valid data available after cleaning. Try a more recent date.")
                st.stop()
            
            if len(data) < 2:
                st.error("Not enough data points. Please select an earlier buy date.")
                st.stop()

        # Calculate portfolio performance
        portfolio_values = []
        dates = data.index
        
        # Get initial prices (first day Close prices)
        initial_prices = data.iloc[0]
        
        # Calculate portfolio value for each date
        for date_idx in range(len(data)):
            current_prices = data.iloc[date_idx]
            portfolio_value = 0
            
            for ticker in data.columns:
                if ticker in weights:
                    weight_pct = weights[ticker] / 100
                    initial_allocation = initial_investment * weight_pct
                    
                    if initial_prices[ticker] > 0:  # Avoid division by zero
                        shares = initial_allocation / initial_prices[ticker]
                        current_value = shares * current_prices[ticker]
                        portfolio_value += current_value
            
            portfolio_values.append(portfolio_value)
        
        # Create performance DataFrame
        performance_df = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': portfolio_values
        })
        
        # Calculate returns
        performance_df['Daily_Return'] = performance_df['Portfolio_Value'].pct_change()
        performance_df['Cumulative_Return'] = (performance_df['Portfolio_Value'] / initial_investment - 1) * 100
        
        # Get benchmark (S&P 500) for comparison
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(start=buy_date, end=date.today(), auto_adjust=True)
            if not spy_hist.empty and 'Close' in spy_hist.columns:
                spy_data = spy_hist['Close']
                spy_returns = (spy_data / spy_data.iloc[0] - 1) * 100
            else:
                spy_returns = pd.Series()
        except:
            spy_returns = pd.Series()
            st.info("Could not fetch S&P 500 data for comparison")
        
        # Calculate metrics
        current_value = portfolio_values[-1]
        total_return = (current_value / initial_investment - 1) * 100
        
        # Calculate days held
        calendar_days = (dates[-1] - dates[0]).days
        trading_days = len(performance_df)
        
        # Calculate annualized return
        years_held = calendar_days / 365.25
        if years_held > 0:
            annualized_return = ((current_value / initial_investment) ** (1/years_held) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate volatility
        portfolio_volatility = performance_df['Daily_Return'].std() * np.sqrt(252) * 100
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Portfolio Value",
                f"${current_value:,.2f}",
                f"${current_value - initial_investment:,.2f}"
            )
        
        with col2:
            st.metric(
                "Total Return",
                f"{total_return:.2f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "Annualized Return",
                f"{annualized_return:.2f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "Annualized Volatility",
                f"{portfolio_volatility:.2f}%",
                delta=None
            )
        
        # Charts
        st.subheader("Portfolio Performance Over Time")
        
        # Portfolio value chart
        fig_value = go.Figure()
        fig_value.add_trace(go.Scatter(
            x=performance_df['Date'],
            y=performance_df['Portfolio_Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_value.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_value, use_container_width=True)
        
        # Returns comparison chart
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(
            x=performance_df['Date'],
            y=performance_df['Cumulative_Return'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add S&P 500 if available
        if not spy_returns.empty and len(spy_returns) > 0:
            # Align dates
            spy_aligned = spy_returns.reindex(performance_df['Date'], method='ffill')
            fig_returns.add_trace(go.Scatter(
                x=performance_df['Date'],
                y=spy_aligned.values,
                mode='lines',
                name='S&P 500 (SPY)',
                line=dict(color='#ff7f0e', width=2)
            ))
        
        fig_returns.update_layout(
            title="Cumulative Returns: Portfolio vs S&P 500",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Portfolio allocation pie chart
        st.subheader("Current Portfolio Allocation")
        
        # Only show tickers that are in the data
        active_weights = {k: v for k, v in weights.items() if k in data.columns and v > 0}
        
        if active_weights:
            fig_pie = px.pie(
                values=list(active_weights.values()),
                names=list(active_weights.keys()),
                title="Portfolio Weights (Active Holdings)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Calculate maximum drawdown
        cummax = performance_df['Portfolio_Value'].cummax()
        drawdown = (performance_df['Portfolio_Value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Calculate Sharpe Ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_return = annualized_return - (risk_free_rate * 100)
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Performance statistics table
        st.subheader("Performance Statistics")
        stats_data = {
            "Metric": [
                "Initial Investment",
                "Current Value",
                "Total Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio (RF=2%)",
                "Maximum Drawdown",
                "Days Held"
            ],
            "Value": [
                f"${initial_investment:,.2f}",
                f"${current_value:,.2f}",
                f"{total_return:.2f}%",
                f"{annualized_return:.2f}%",
                f"{portfolio_volatility:.2f}%",
                f"{sharpe_ratio:.3f}",
                f"{max_drawdown:.2f}%",
                f"{calendar_days} days ({trading_days} trading days)"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Individual stock performance
        st.subheader("Individual Stock Performance")
        stock_perf = []
        for ticker in data.columns:
            if ticker in weights and weights[ticker] > 0:
                initial_price = data[ticker].iloc[0]
                current_price = data[ticker].iloc[-1]
                stock_return = (current_price / initial_price - 1) * 100
                stock_perf.append({
                    'Ticker': ticker,
                    'Weight (%)': f"{weights[ticker]:.1f}",
                    'Initial Price': f"${initial_price:.2f}",
                    'Current Price': f"${current_price:.2f}",
                    'Return (%)': f"{stock_return:.2f}"
                })
        
        if stock_perf:
            stock_perf_df = pd.DataFrame(stock_perf)
            st.dataframe(stock_perf_df, use_container_width=True, hide_index=True)
        
        # Recent performance table
        st.subheader("Recent Performance (Last 10 Days)")
        if len(performance_df) > 0:
            recent_df = performance_df[['Date', 'Portfolio_Value', 'Daily_Return', 'Cumulative_Return']].tail(10).copy()
            recent_df['Daily_Return'] = recent_df['Daily_Return'].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
            recent_df['Cumulative_Return'] = recent_df['Cumulative_Return'].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            recent_df['Portfolio_Value'] = recent_df['Portfolio_Value'].map(lambda x: f"${x:,.2f}")
            recent_df['Date'] = pd.to_datetime(recent_df['Date']).dt.strftime('%Y-%m-%d')
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Debug info: Please try adjusting your date range or checking if all tickers are valid.")
        # Show debug information
        with st.expander("Debug Information"):
            st.write(f"Active tickers: {[t for t in default_tickers if weights[t] > 0]}")
            st.write(f"Buy date: {buy_date}")
            st.write(f"Error details: {str(e)}")

else:
    st.info("👈 Click 'Update Portfolio Performance' to load current data")
    
    # Show portfolio info
    st.subheader("📊 Portfolio Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Selected Buy Date:** {buy_date.strftime('%Y-%m-%d')}  
        **Initial Investment:** ${initial_investment:,}  
        **Days Since Buy Date:** {(date.today() - buy_date).days} days  
        **Status:** Ready to track performance
        """)
    
    with col2:
        # Show current portfolio allocation
        allocation_df = pd.DataFrame({
            'Ticker': default_tickers,
            'Weight (%)': [f"{weights[ticker]:.1f}" for ticker in default_tickers],
            'Company': ['Howmet Aerospace', 'NVIDIA', 'Motorola Solutions', 
                       'Amazon', 'Mastercard', 'Tesla', 'Albemarle Corp']
        })
        # Only show tickers with non-zero weights
        allocation_df = allocation_df[[float(w.replace('%', '')) > 0 for w in allocation_df['Weight (%)'].values]]
        if not allocation_df.empty:
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Please assign weights to at least one ticker.")

# Footer
st.markdown("---")
st.markdown("**Note:** This dashboard uses Yahoo Finance data. Past performance does not guarantee future results.")