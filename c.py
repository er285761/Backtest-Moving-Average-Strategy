import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import wikipedia
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ta 
from newsapi import NewsApiClient
import time
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Function to backtest the Moving Average Crossover strategy
def backtest_ma_crossover_strategy(df, short_ma_window, long_ma_window, initial_capital, transaction_cost_percentage, slippage_percentage):
    if 'Close' not in df.columns:
        st.error("Error: 'Close' column not found in DataFrame.")
        return

    # Calculate Moving Averages
    df['Short_MA'] = df['Close'].rolling(window=short_ma_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_ma_window).mean()

    # Initialize additional columns for backtesting with transaction costs and slippage
    df['Transaction_Cost'] = 0
    df['Slippage'] = 0

    # Calculate backtested positions without transaction costs and slippage
    df['Position'] = 0
    df.loc[df.index[long_ma_window:], 'Position'] = np.where(df['Short_MA'][long_ma_window:] > df['Long_MA'][long_ma_window:], 1,
                                                         np.where(df['Short_MA'][long_ma_window:] < df['Long_MA'][long_ma_window:], -1, 0))

    df['Strategy'] = df['Position'].shift(1)
    st.write(df['Strategy'])

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate additional backtesting metrics without transaction costs and slippage
    df['Strategy_Return'] = df['Daily_Return'] * df['Strategy']

    # Calculate transaction costs and slippage
    df['Transaction_Cost'] = transaction_cost_percentage * abs(df['Position'].diff())
    df['Slippage'] = slippage_percentage * df['Close'] * abs(df['Position'].diff())

    # Calculate cumulative transaction costs and slippage
    df['Cumulative_Transaction_Cost'] = df['Transaction_Cost'].cumsum()
    df['Cumulative_Slippage'] = df['Slippage'].cumsum()

    # Calculate total trading costs
    total_transaction_costs = df['Transaction_Cost'].sum()
    total_slippage = df['Slippage'].sum()
    total_trading_costs = total_transaction_costs + total_slippage

    # Calculate remaining capital after all trades with trading costs
    remaining_capital = initial_capital - total_trading_costs

    # Calculate cumulative returns and strategy returns with transaction costs and slippage
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1

    # Calculate trading metrics
    total_winning_trades = (df['Strategy_Return'] > 0).sum()
    total_losing_trades = (df['Strategy_Return'] < 0).sum()
    total_winning_trade_value = df.loc[df['Strategy_Return'] > 0, 'Strategy_Return'].sum()
    total_losing_trade_value = df.loc[df['Strategy_Return'] < 0, 'Strategy_Return'].sum()

    win_loss_ratio = total_winning_trades / total_losing_trades if total_losing_trades != 0 else float('inf')
    profit_factor = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')
    risk_reward_ratio = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')

    # Calculate the number of shares bought, sold, or owned
    df['Shares_Bought'] = 0
    df['Shares_Sold'] = 0
    df['Shares_Owned'] = 0
    initial_price = df['Close'].iloc[0]

    if initial_price > 0:
        short_ma = df['Short_MA']
        long_ma = df['Long_MA']

        # Condition for buying based on moving average crossover
        buy_condition = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))

        # Condition for selling based on moving average crossover
        sell_condition = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

        # Number of shares bought is proportional to the remaining capital and current stock price
        df.loc[buy_condition, 'Shares_Bought'] = remaining_capital / initial_price

        # Number of shares sold is proportional to the remaining owned shares
        df.loc[sell_condition, 'Shares_Sold'] = df['Shares_Owned'] * slippage_percentage

        # Update the total owned shares
        df['Shares_Owned'] = df['Shares_Bought'].cumsum() - df['Shares_Sold'].cumsum()
        st.write(df['Shares_Owned'])

        shares_bought = df.loc[buy_condition, 'Shares_Bought']

    # Calculate profits or losses based on the strategy
    current_price = df['Close'].iloc[-1]
    df['Profits_Losses'] = df['Shares_Owned'] * current_price
    st.write(df['Profits_Losses'])

    # Calculate cumulative profits and losses
    df['Cumulative_Profits_Losses'] = df['Profits_Losses'].cumsum()

    # Calculate the profit or loss
    profit_loss = df['Profits_Losses'].iloc[-1] - (initial_price * df['Shares_Owned'].iloc[-1])

    # Determine Buy, Sell, and Hold signals
    df['Signal'] = np.where(df['Short_MA'] > df['Long_MA'], 'Buy', 'Sell')
    df['Signal'] = np.where((df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) > df['Long_MA'].shift(1)), 'Hold', df['Signal'])

    # Display the combined chart with moving averages and Buy/Sell signals
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    ax_combined.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax_combined.plot(df.index, df['Short_MA'], label=f'{short_ma_window}MA', linestyle='--', color='grey')
    ax_combined.plot(df.index, df['Long_MA'], label=f'{long_ma_window}MA', linestyle='--', color='lightgrey')

    # Plot Buy signals with green arrow up
    ax_combined.scatter(df.index[df['Signal'] == 'Buy'], df['Close'][df['Signal'] == 'Buy'], marker='^', color='g', label='Buy Signal')

    # Plot Sell signals with red arrow down
    ax_combined.scatter(df.index[df['Signal'] == 'Sell'], df['Close'][df['Signal'] == 'Sell'], marker='v', color='r', label='Sell Signal')

    ax_combined.set_ylabel('Price ($)')
    ax_combined.legend(loc='upper left')
    ax_combined.xaxis.set_major_locator(mdates.MonthLocator())

    # Display the combined chart
    st.pyplot(fig_combined)

    # Determine buy/sell signals based on moving averages
    if df['Short_MA'].iloc[-1] > df['Long_MA'].iloc[-1] and df['Short_MA'].iloc[-2] <= df['Long_MA'].iloc[-2]:
        st.write(f"**Buy Signal:** Golden Cross - {short_ma_window}-day MA crossed above {long_ma_window}-day MA. Consider buying {ticker_symbol}.")
    elif df['Short_MA'].iloc[-1] < df['Long_MA'].iloc[-1] and df['Short_MA'].iloc[-2] >= df['Long_MA'].iloc[-2]:
        st.write(f"**Sell Signal:** Death Cross - {short_ma_window}-day MA crossed below {long_ma_window}-day MA. Consider selling {ticker_symbol}.")
    else:
        st.write(f"**Hold Signal:** No significant MA crossover detected. Hold {ticker_symbol}.")

    # Display cumulative profits and losses in the plot
    fig_cumulative_pl, ax_cumulative_pl = plt.subplots(figsize=(10, 6))
    ax_cumulative_pl.plot(df.index, df['Cumulative_Profits_Losses'], label='Cumulative Profits/Losses', color='blue')
    ax_cumulative_pl.set_title('Cumulative Profits and Losses')
    ax_cumulative_pl.set_xlabel('Date')
    ax_cumulative_pl.set_ylabel('Cumulative Profits/Losses')
    ax_cumulative_pl.legend()
    st.pyplot(fig_cumulative_pl)

    # Display cumulative returns with trading costs in the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Cumulative_Return'], label='Cumulative Returns', color='green')
    ax.plot(df.index, df['Cumulative_Strategy_Return'], label='Cumulative Strategy Returns with Costs', color='orange')
    ax.set_title('Backtesting Results for Moving Average Crossover Strategy with Costs')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    st.pyplot(fig)

    # Display trading metrics with transaction costs and slippage
    st.write("## Trading Metrics with Costs:")
    st.write(f"Total Winning Trades: {total_winning_trades}")
    st.write(f"Total Losing Trades: {total_losing_trades}")
    st.write(f"Total Winning Trade Value: ${total_winning_trade_value:.2f}")
    st.write(f"Total Losing Trade Value: ${total_losing_trade_value:.2f}")
    st.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    st.write(f"Profit Factor: {profit_factor:.2f}")
    st.write(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}")

    st.write("## Total Value of Stock:")
    st.write(f"Total Value of Stock: ${df['Profits_Losses'].iloc[-1]:,.2f}")
    st.write(f"Profit or Loss: ${profit_loss.sum():,.2f}")

    # Display total trading costs
    st.write("## Total Trading Costs:")
    st.write(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
    st.write(f"Total Slippage: ${total_slippage:.2f}")
    st.write(f"Total Trading Costs: ${total_trading_costs:.2f}")

    # Display the number of shares bought or owned
    st.write("## Number of Shares Bought or Owned:")
    st.write(f"Shares Bought: {shares_bought.iloc[0]:.2f}")
    st.write(f"Shares Sold: {df['Shares_Sold'].sum():.2f}")
    st.write(f"Shares Owned: {df['Shares_Owned'].iloc[-1]:.2f}")
    
    # Display remaining capital with trading costs
    remaining_capital_message = f"Remaining Capital with Costs: ${remaining_capital:.2f}"
    st.write(remaining_capital_message)
    st.success("Backtesting completed successfully!")

# Set Streamlit app title
st.write("""# Online Stock Price Ticker""")

# User input for ticker symbol
ticker_symbol = st.text_input("Enter Ticker Symbol:", 'TSLA')  # Default to 'TSLA'

# User input for start date
start_date = st.text_input("Enter Start Date (YYYY-MM-DD):", '2021-01-01')

tickerData = yf.Ticker(ticker_symbol)
tickerDf = tickerData.history(period='1d', start=start_date, end=pd.to_datetime('today'))
tickerDf.isnull().sum()

st.write(f"### Historical Stock Prices of {ticker_symbol}")
st.write(tickerDf)

# Display stock price and volume using line charts
st.write(f"### Historical Price Chart of {ticker_symbol}")
st.line_chart(tickerDf.Close)

st.write(f"### Historical Volume Trend of {ticker_symbol}")
st.line_chart(tickerDf.Volume)

# User input for initial capital
initial_capital = st.number_input("Enter Initial Capital ($):", value=10000.0, step=1.0)

st.write(f"### Calculate Simple Moving Averages of {ticker_symbol}")
# User input for moving average window sizes
short_ma_window = st.number_input("Enter Short-term Moving Average Window (e.g., 12):", value=12, step=1)
long_ma_window = st.number_input("Enter Long-term Moving Average Window (e.g., 26):", value=2, step=1)

# Example usage
backtest_ma_crossover_strategy(tickerDf, short_ma_window, long_ma_window, initial_capital, transaction_cost_percentage=0.01, slippage_percentage=0.005)

# Function to backtest the Exponential Moving Average (EMA) Crossover strategy
def backtest_ema_crossover_strategy(df, short_ema_span, long_ema_span, initial_capital, transaction_cost_percentage, slippage_percentage):
    if 'Close' not in df.columns:
        st.error("Error: 'Close' column not found in DataFrame.")
        return

    # Calculate Exponential Moving Averages (EMAs)
    df['Short_EMA'] = df['Close'].ewm(span=short_ema_span, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_ema_span, adjust=False).mean()

    # Initialize additional columns for backtesting with transaction costs and slippage
    df['Transaction_Cost'] = 0
    df['Slippage'] = 0

    # Calculate backtested positions without transaction costs and slippage
    df['Position'] = 0
    df.loc[df.index[long_ema_span:], 'Position'] = np.where(df['Short_EMA'][long_ema_span:] > df['Long_EMA'][long_ema_span:], 1,
                                                            np.where(df['Short_EMA'][long_ema_span:] < df['Long_EMA'][long_ema_span:], -1, 0))

    df['Strategy'] = df['Position'].shift(1)
    st.write(df['Strategy'])

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate additional backtesting metrics without transaction costs and slippage
    df['Strategy_Return'] = df['Daily_Return'] * df['Strategy']

    # Calculate transaction costs and slippage
    df['Transaction_Cost'] = transaction_cost_percentage * abs(df['Position'].diff())
    df['Slippage'] = slippage_percentage * df['Close'] * abs(df['Position'].diff())

    # Calculate cumulative transaction costs and slippage
    df['Cumulative_Transaction_Cost'] = df['Transaction_Cost'].cumsum()
    df['Cumulative_Slippage'] = df['Slippage'].cumsum()

    # Calculate total trading costs
    total_transaction_costs = df['Transaction_Cost'].sum()
    total_slippage = df['Slippage'].sum()
    total_trading_costs = total_transaction_costs + total_slippage

    # Calculate remaining capital after all trades with trading costs
    remaining_capital = initial_capital - total_trading_costs

    # Calculate cumulative returns and strategy returns with transaction costs and slippage
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1

    # Calculate trading metrics
    total_winning_trades = (df['Strategy_Return'] > 0).sum()
    total_losing_trades = (df['Strategy_Return'] < 0).sum()
    total_winning_trade_value = df.loc[df['Strategy_Return'] > 0, 'Strategy_Return'].sum()
    total_losing_trade_value = df.loc[df['Strategy_Return'] < 0, 'Strategy_Return'].sum()

    win_loss_ratio = total_winning_trades / total_losing_trades if total_losing_trades != 0 else float('inf')
    profit_factor = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')
    risk_reward_ratio = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')

    # Calculate the number of shares bought, sold, or owned
    df['Shares_Bought'] = 0
    df['Shares_Sold'] = 0
    df['Shares_Owned'] = 0
    initial_price = df['Close'].iloc[0]

    if initial_price > 0:
        short_ema = df['Short_EMA']
        long_ema = df['Long_EMA']

        # Condition for buying based on exponential moving average crossover
        buy_condition = (short_ema > long_ema) & (short_ema.shift(1) <= long_ema.shift(1))

        # Condition for selling based on exponential moving average crossover
        sell_condition = (short_ema < long_ema) & (short_ema.shift(1) >= long_ema.shift(1))

        # Number of shares bought is proportional to the remaining capital and current stock price
        df.loc[buy_condition, 'Shares_Bought'] = remaining_capital / initial_price

        # Number of shares sold is proportional to the remaining owned shares
        df.loc[sell_condition, 'Shares_Sold'] = df['Shares_Owned'] * slippage_percentage

        # Update the total owned shares
        df['Shares_Owned'] = df['Shares_Bought'].cumsum() - df['Shares_Sold'].cumsum()

        shares_bought = df.loc[buy_condition, 'Shares_Bought']

    # Calculate profits or losses based on the strategy
    current_price = df['Close'].iloc[-1]
    df['Profits_Losses'] = df['Shares_Owned'] * current_price

    # Calculate cumulative profits and losses
    df['Cumulative_Profits_Losses'] = df['Profits_Losses'].cumsum()

    # Calculate the profit or loss
    profit_loss = df['Profits_Losses'].iloc[-1] - (initial_price * df['Shares_Owned'].iloc[-1])

    # Calculate the number of shares sold
    df['Shares_Sold'] = df['Shares_Owned'].diff().clip(upper=0) * -1

    # Determine Buy, Sell, and Hold signals
    df['Signal'] = np.where(df['Short_EMA'] > df['Long_EMA'], 'Buy', 'Sell')
    df['Signal'] = np.where((df['Short_EMA'] > df['Long_EMA']) & (df['Short_EMA'].shift(1) > df['Long_EMA'].shift(1)), 'Hold', df['Signal'])

    # Display the combined chart with moving averages and Buy/Sell signals
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    ax_combined.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax_combined.plot(df.index, df['Short_EMA'], label=f'{short_ema_span}EMA', linestyle='--', color='grey')
    ax_combined.plot(df.index, df['Long_EMA'], label=f'{long_ema_span}EMA', linestyle='--', color='lightgrey')

    # Plot Buy signals with green arrow up
    ax_combined.scatter(df.index[df['Signal'] == 'Buy'], df['Close'][df['Signal'] == 'Buy'], marker='^', color='g', label='Buy Signal')

    # Plot Sell signals with red arrow down
    ax_combined.scatter(df.index[df['Signal'] == 'Sell'], df['Close'][df['Signal'] == 'Sell'], marker='v', color='r', label='Sell Signal')

    ax_combined.set_ylabel('Price ($)')
    ax_combined.legend(loc='upper left')
    ax_combined.xaxis.set_major_locator(mdates.MonthLocator())

    # Display the combined chart
    st.pyplot(fig_combined)

    # Determine buy/sell signals based on exponential moving averages
    if df['Short_EMA'].iloc[-1] > df['Long_EMA'].iloc[-1] and df['Short_EMA'].iloc[-2] <= df['Long_EMA'].iloc[-2]:
        st.write(f"**Buy Signal:** Golden Cross - {short_ema_span}-day EMA crossed above {long_ema_span}-day EMA. Consider buying {ticker_symbol}.")
    elif df['Short_EMA'].iloc[-1] < df['Long_EMA'].iloc[-1] and df['Short_EMA'].iloc[-2] >= df['Long_EMA'].iloc[-2]:
        st.write(f"**Sell Signal:** Death Cross - {short_ema_span}-day EMA crossed below {long_ema_span}-day EMA. Consider selling {ticker_symbol}.")
    else:
        st.write(f"**Hold Signal:** No significant EMA crossover detected. Hold {ticker_symbol}.")

    # Display cumulative profits and losses in the plot
    fig_cumulative_pl, ax_cumulative_pl = plt.subplots(figsize=(10, 6))
    ax_cumulative_pl.plot(df.index, df['Cumulative_Profits_Losses'], label='Cumulative Profits/Losses', color='blue')
    ax_cumulative_pl.set_title('Cumulative Profits and Losses')
    ax_cumulative_pl.set_xlabel('Date')
    ax_cumulative_pl.set_ylabel('Cumulative Profits/Losses')
    ax_cumulative_pl.legend()
    st.pyplot(fig_cumulative_pl)

    # Display cumulative returns with trading costs in the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Cumulative_Return'], label='Cumulative Returns', color='green')
    ax.plot(df.index, df['Cumulative_Strategy_Return'], label='Cumulative Strategy Returns with Costs', color='orange')
    ax.set_title('Backtesting Results for Exponential Moving Average Crossover Strategy with Costs')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    st.pyplot(fig)

    # Display trading metrics with transaction costs and slippage
    st.write("## Trading Metrics with Costs:")
    st.write(f"Total Winning Trades: {total_winning_trades}")
    st.write(f"Total Losing Trades: {total_losing_trades}")
    st.write(f"Total Winning Trade Value: ${total_winning_trade_value:.2f}")
    st.write(f"Total Losing Trade Value: ${total_losing_trade_value:.2f}")
    st.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    st.write(f"Profit Factor: {profit_factor:.2f}")
    st.write(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}")

    st.write("## Total Value of Stock:")
    st.write(f"Total Value of Stock: ${df['Profits_Losses'].iloc[-1]:,.2f}")
    st.write(f"Profit or Loss: ${profit_loss.sum():,.2f}")

    # Display total trading costs
    st.write("## Total Trading Costs:")
    st.write(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
    st.write(f"Total Slippage: ${total_slippage:.2f}")
    st.write(f"Total Trading Costs: ${total_trading_costs:.2f}")
    
    # Display the number of shares bought or owned
    st.write("## Number of Shares Bought or Owned:")
    st.write(f"Shares Bought: {shares_bought.iloc[0]:.2f}")
    st.write(f"Shares Owned: {df['Shares_Owned'].iloc[-1]:.2f}")
   
    
    # Display remaining capital with trading costs
    remaining_capital_message = f"Remaining Capital with Costs: ${remaining_capital:.2f}"
    st.write(remaining_capital_message)
    st.success("Backtesting completed successfully!")
    

st.write(f"### Calculate Exponential Moving Averages of {ticker_symbol}")
# User input for exponential moving average span
short_ema_span = st.number_input("Enter Short-term Exponential Moving Average Span (e.g., 12):", value=12, step=1)
long_ema_span = st.number_input("Enter Long-term Exponential Moving Average Span (e.g., 26):", value=26, step=1)

# Example usage
backtest_ema_crossover_strategy(tickerDf, short_ema_span, long_ema_span, initial_capital, transaction_cost_percentage=0.01, slippage_percentage=0.005)

# Function to backtest the Moving Average Convergence Divergence (MACD) strategy
def backtest_macd_strategy(df, short_macd_span, long_macd_span, signal_macd_span, initial_capital, transaction_cost_percentage, slippage_percentage):
    if 'Close' not in df.columns:
        st.error("Error: 'Close' column not found in DataFrame.")
        return

    # Calculate Exponential Moving Averages (EMAs) for MACD
    df['Short_EMA'] = df['Close'].ewm(span=short_macd_span, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_macd_span, adjust=False).mean()

    # Calculate MACD line
    df['MACD'] = df['Short_EMA'] - df['Long_EMA']

    # Calculate Signal line (EMA of MACD)
    df['Signal_Line'] = df['MACD'].ewm(span=signal_macd_span, adjust=False).mean()

    # Initialize additional columns for backtesting with transaction costs and slippage
    df['Transaction_Cost'] = 0
    df['Slippage'] = 0

    # Calculate backtested positions without transaction costs and slippage
    df['Position'] = 0
    df.loc[df.index[long_macd_span:], 'Position'] = np.where(df['MACD'][long_macd_span:] > df['Signal_Line'][long_macd_span:], 1,
                                                            np.where(df['MACD'][long_macd_span:] < df['Signal_Line'][long_macd_span:], -1, 0))

    df['Strategy'] = df['Position'].shift(1)
    st.write(df['Strategy'])

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate additional backtesting metrics without transaction costs and slippage
    df['Strategy_Return'] = df['Daily_Return'] * df['Strategy']

    # Calculate transaction costs and slippage
    df['Transaction_Cost'] = transaction_cost_percentage * abs(df['Position'].diff())
    df['Slippage'] = slippage_percentage * df['Close'] * abs(df['Position'].diff())

    # Calculate cumulative transaction costs and slippage
    df['Cumulative_Transaction_Cost'] = df['Transaction_Cost'].cumsum()
    df['Cumulative_Slippage'] = df['Slippage'].cumsum()

    # Calculate total trading costs
    total_transaction_costs = df['Transaction_Cost'].sum()
    total_slippage = df['Slippage'].sum()
    total_trading_costs = total_transaction_costs + total_slippage

    # Calculate remaining capital after all trades with trading costs
    remaining_capital = initial_capital - total_trading_costs

    # Calculate cumulative returns and strategy returns with transaction costs and slippage
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1

    # Calculate trading metrics
    total_winning_trades = (df['Strategy_Return'] > 0).sum()
    total_losing_trades = (df['Strategy_Return'] < 0).sum()
    total_winning_trade_value = df.loc[df['Strategy_Return'] > 0, 'Strategy_Return'].sum()
    total_losing_trade_value = df.loc[df['Strategy_Return'] < 0, 'Strategy_Return'].sum()

    win_loss_ratio = total_winning_trades / total_losing_trades if total_losing_trades != 0 else float('inf')
    profit_factor = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')
    risk_reward_ratio = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')

    # Calculate the number of shares bought, sold, or owned
    df['Shares_Bought'] = 0
    df['Shares_Sold'] = 0
    df['Shares_Owned'] = 0
    initial_price = df['Close'].iloc[0]

    if initial_price > 0:
        macd = df['MACD']
        signal_line = df['Signal_Line']

        # Condition for buying based on MACD crossover
        buy_condition = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))

        # Condition for selling based on MACD crossover
        sell_condition = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))

        # Number of shares bought is proportional to the remaining capital and current stock price
        df.loc[buy_condition, 'Shares_Bought'] = remaining_capital / initial_price

        # Number of shares sold is proportional to the remaining owned shares
        df.loc[sell_condition, 'Shares_Sold'] = df['Shares_Owned'] * slippage_percentage

        # Update the total owned shares
        df['Shares_Owned'] = df['Shares_Bought'].cumsum() - df['Shares_Sold'].cumsum()

        shares_bought = df.loc[buy_condition, 'Shares_Bought']

    # Calculate profits or losses based on the strategy
    current_price = df['Close'].iloc[-1]
    df['Profits_Losses'] = df['Shares_Owned'] * current_price

    # Calculate cumulative profits and losses
    df['Cumulative_Profits_Losses'] = df['Profits_Losses'].cumsum()

    # Calculate the profit or loss
    profit_loss = df['Profits_Losses'].iloc[-1] - (initial_price * df['Shares_Owned'].iloc[-1])

    # Determine Buy, Sell, and Hold signals
    df['Signal'] = np.where(df['MACD'] > df['Signal_Line'], 'Buy', 'Sell')
    df['Signal'] = np.where((df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) > df['Signal_Line'].shift(1)), 'Hold', df['Signal'])

    # Display the combined chart with MACD and Signal line
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    ax_combined.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax_combined.plot(df.index, df['MACD'], label=f'MACD ({short_macd_span}-{long_macd_span})', linestyle='--', color='blue')
    ax_combined.plot(df.index, df['Signal_Line'], label=f'Signal Line ({signal_macd_span})', linestyle='--', color='red')

    # Plot Buy signals with green arrow up
    ax_combined.scatter(df.index[df['Signal'] == 'Buy'], df['Close'][df['Signal'] == 'Buy'], marker='^', color='g', label='Buy Signal')

    # Plot Sell signals with red arrow down
    ax_combined.scatter(df.index[df['Signal'] == 'Sell'], df['Close'][df['Signal'] == 'Sell'], marker='v', color='r', label='Sell Signal')

    ax_combined.set_ylabel('Price ($)')
    ax_combined.legend(loc='upper left')
    ax_combined.xaxis.set_major_locator(mdates.MonthLocator())

    # Display the combined chart
    st.pyplot(fig_combined)

    # Determine buy/sell signals based on MACD
    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
        st.write(f"**Buy Signal:** MACD crossed above Signal Line. Consider buying {ticker_symbol}.")
    elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]:
        st.write(f"**Sell Signal:** MACD crossed below Signal Line. Consider selling {ticker_symbol}.")
    else:
        st.write(f"**Hold Signal:** No significant MACD crossover detected. Hold {ticker_symbol}.")

    # Display cumulative profits and losses in the plot
    fig_cumulative_pl, ax_cumulative_pl = plt.subplots(figsize=(10, 6))
    ax_cumulative_pl.plot(df.index, df['Cumulative_Profits_Losses'], label='Cumulative Profits/Losses', color='blue')
    ax_cumulative_pl.set_title('Cumulative Profits and Losses')
    ax_cumulative_pl.set_xlabel('Date')
    ax_cumulative_pl.set_ylabel('Cumulative Profits/Losses')
    ax_cumulative_pl.legend()
    st.pyplot(fig_cumulative_pl)

    # Display cumulative returns with trading costs in the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Cumulative_Return'], label='Cumulative Returns', color='green')
    ax.plot(df.index, df['Cumulative_Strategy_Return'], label='Cumulative Strategy Returns with Costs', color='orange')
    ax.set_title('Backtesting Results for MACD Strategy with Costs')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    st.pyplot(fig)

    # Display trading metrics with transaction costs and slippage
    st.write("## Trading Metrics with Costs:")
    st.write(f"Total Winning Trades: {total_winning_trades}")
    st.write(f"Total Losing Trades: {total_losing_trades}")
    st.write(f"Total Winning Trade Value: ${total_winning_trade_value:.2f}")
    st.write(f"Total Losing Trade Value: ${total_losing_trade_value:.2f}")
    st.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    st.write(f"Profit Factor: {profit_factor:.2f}")
    st.write(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}")

    st.write("## Total Value of Stock:")
    st.write(f"Total Value of Stock: ${df['Profits_Losses'].iloc[-1]:,.2f}")
    st.write(f"Profit or Loss: ${profit_loss.sum():,.2f}")

    # Display total trading costs
    st.write("## Total Trading Costs:")
    st.write(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
    st.write(f"Total Slippage: ${total_slippage:.2f}")
    st.write(f"Total Trading Costs: ${total_trading_costs:.2f}")

    # Display the number of shares bought or owned
    st.write("## Number of Shares Bought or Owned:")
    st.write(f"Shares Bought: {shares_bought.iloc[0]:.2f}")
    st.write(f"Shares Sold: {df['Shares_Sold'].sum():.2f}")
    st.write(f"Shares Owned: {df['Shares_Owned'].iloc[-1]:.2f}")

    # Display remaining capital with trading costs
    remaining_capital_message = f"Remaining Capital with Costs: ${remaining_capital:.2f}"
    st.write(remaining_capital_message)
    st.success("Backtesting completed successfully!")

st.write(f"### Moving Average Convergence Divergence {ticker_symbol}")
# User input for moving average window sizes
short_macd_span = st.number_input("Enter Short-term MACD Span (e.g., 12):", value=12, step=1)
long_macd_span = st.number_input("Enter Long-term MACD Span (e.g., 26):", value=26, step=1)
signal_macd_span = st.number_input("Enter Signal Line MACD Span (e.g., 9):", value=9, step=1)

# Example usage
backtest_macd_strategy(tickerDf, short_macd_span, long_macd_span, signal_macd_span, initial_capital, transaction_cost_percentage=0.01, slippage_percentage=0.005)

# Function to backtest the Stochastic RSI strategy
def backtest_stochastic_rsi_strategy(df, rsi_window, stoch_window, overbought_threshold, oversold_threshold, initial_capital, transaction_cost_percentage, slippage_percentage):
    if 'Close' not in df.columns:
        st.error("Error: 'Close' column not found in DataFrame.")
        return

    # Calculate RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_window).rsi()

    # Calculate Stochastic RSI
    df['Stoch_RSI'] = (df['RSI'] - df['RSI'].rolling(window=stoch_window).min()) / (
            df['RSI'].rolling(window=stoch_window).max() - df['RSI'].rolling(window=stoch_window).min())

    # Initialize additional columns for backtesting with transaction costs and slippage
    df['Transaction_Cost'] = 0
    df['Slippage'] = 0

    # Calculate backtested positions without transaction costs and slippage
    df['Position'] = 0
    df.loc[df.index[stoch_window:], 'Position'] = np.where(
        (df['Stoch_RSI'][stoch_window:] > overbought_threshold) & (df['Stoch_RSI'][stoch_window:].shift(1) <= overbought_threshold), -1,
        np.where((df['Stoch_RSI'][stoch_window:] < oversold_threshold) & (df['Stoch_RSI'][stoch_window:].shift(1) >= oversold_threshold), 1, 0))

    df['Strategy'] = df['Position'].shift(1)
    st.write(df['Strategy'])

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate additional backtesting metrics without transaction costs and slippage
    df['Strategy_Return'] = df['Daily_Return'] * df['Strategy']

    # Calculate transaction costs and slippage
    df['Transaction_Cost'] = transaction_cost_percentage * abs(df['Position'].diff())
    df['Slippage'] = slippage_percentage * df['Close'] * abs(df['Position'].diff())

    # Calculate cumulative transaction costs and slippage
    df['Cumulative_Transaction_Cost'] = df['Transaction_Cost'].cumsum()
    df['Cumulative_Slippage'] = df['Slippage'].cumsum()

    # Calculate total trading costs
    total_transaction_costs = df['Transaction_Cost'].sum()
    total_slippage = df['Slippage'].sum()
    total_trading_costs = total_transaction_costs + total_slippage

    # Calculate remaining capital after all trades with trading costs
    remaining_capital = initial_capital - total_trading_costs

    # Calculate cumulative returns and strategy returns with transaction costs and slippage
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1

    # Calculate trading metrics
    total_winning_trades = (df['Strategy_Return'] > 0).sum()
    total_losing_trades = (df['Strategy_Return'] < 0).sum()
    total_winning_trade_value = df.loc[df['Strategy_Return'] > 0, 'Strategy_Return'].sum()
    total_losing_trade_value = df.loc[df['Strategy_Return'] < 0, 'Strategy_Return'].sum()

    win_loss_ratio = total_winning_trades / total_losing_trades if total_losing_trades != 0 else float('inf')
    profit_factor = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')
    risk_reward_ratio = total_winning_trade_value / abs(total_losing_trade_value) if total_losing_trade_value != 0 else float('inf')

    # Calculate the number of shares bought, sold, or owned
    df['Shares_Bought'] = 0
    df['Shares_Sold'] = 0
    df['Shares_Owned'] = 0
    initial_price = df['Close'].iloc[0]

    if initial_price > 0:
        stoch_rsi = df['Stoch_RSI']

        # Condition for buying based on Stochastic RSI
        buy_condition = (stoch_rsi < oversold_threshold) & (stoch_rsi.shift(1) >= oversold_threshold)

        # Condition for selling based on Stochastic RSI
        sell_condition = (stoch_rsi > overbought_threshold) & (stoch_rsi.shift(1) <= overbought_threshold)

        # Number of shares bought is proportional to the remaining capital and current stock price
        df.loc[buy_condition, 'Shares_Bought'] = remaining_capital / initial_price

        # Number of shares sold is proportional to the remaining owned shares
        df.loc[sell_condition, 'Shares_Sold'] = df['Shares_Owned'] * slippage_percentage

        # Update the total owned shares
        df['Shares_Owned'] = df['Shares_Bought'].cumsum() - df['Shares_Sold'].cumsum()

        shares_bought = df.loc[buy_condition, 'Shares_Bought']

    # Calculate profits or losses based on the strategy
    current_price = df['Close'].iloc[-1]
    df['Profits_Losses'] = df['Shares_Owned'] * current_price
    st.write(df['Profits_Losses'])

    # Calculate cumulative profits and losses
    df['Cumulative_Profits_Losses'] = df['Profits_Losses'].cumsum()

    # Calculate the profit or loss
    profit_loss = df['Profits_Losses'].iloc[-1] - (initial_price * df['Shares_Owned'].iloc[-1])

    # Determine Buy, Sell, and Hold signals
    df['Signal'] = np.where(df['Stoch_RSI'] < oversold_threshold, 'Buy',
                            np.where(df['Stoch_RSI'] > overbought_threshold, 'Sell', 'Hold'))

    # Display the combined chart with Stochastic RSI and Overbought/Oversold signals
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    ax_combined.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax_combined.plot(df.index, df['Stoch_RSI'], label=f'Stochastic RSI ({rsi_window}, {stoch_window})', linestyle='--', color='purple')

    # Plot Buy signals with green arrow up
    ax_combined.scatter(df.index[df['Signal'] == 'Buy'], df['Close'][df['Signal'] == 'Buy'], marker='^', color='g', label='Buy Signal')

    # Plot Sell signals with red arrow down
    ax_combined.scatter(df.index[df['Signal'] == 'Sell'], df['Close'][df['Signal'] == 'Sell'], marker='v', color='r', label='Sell Signal')

    ax_combined.axhline(y=oversold_threshold, color='blue', linestyle='--', label='Oversold Threshold')
    ax_combined.axhline(y=overbought_threshold, color='orange', linestyle='--', label='Overbought Threshold')

    ax_combined.set_ylabel('Price ($)')
    ax_combined.legend(loc='upper left')
    ax_combined.xaxis.set_major_locator(mdates.MonthLocator())

    # Display the combined chart
    st.pyplot(fig_combined)

    # Determine buy/sell signals based on Stochastic RSI
    if df['Stoch_RSI'].iloc[-1] < oversold_threshold and df['Stoch_RSI'].iloc[-2] >= oversold_threshold:
        st.write(f"**Buy Signal:** Stochastic RSI crossed below {oversold_threshold}. Consider buying {ticker_symbol}.")
    elif df['Stoch_RSI'].iloc[-1] > overbought_threshold and df['Stoch_RSI'].iloc[-2] <= overbought_threshold:
        st.write(f"**Sell Signal:** Stochastic RSI crossed above {overbought_threshold}. Consider selling {ticker_symbol}.")
    else:
        st.write(f"**Hold Signal:** No significant Stochastic RSI crossover detected. Hold {ticker_symbol}.")

    # Display cumulative profits and losses in the plot
    fig_cumulative_pl, ax_cumulative_pl = plt.subplots(figsize=(10, 6))
    ax_cumulative_pl.plot(df.index, df['Cumulative_Profits_Losses'], label='Cumulative Profits/Losses', color='blue')
    ax_cumulative_pl.set_title('Cumulative Profits and Losses')
    ax_cumulative_pl.set_xlabel('Date')
    ax_cumulative_pl.set_ylabel('Cumulative Profits/Losses')
    ax_cumulative_pl.legend()
    st.pyplot(fig_cumulative_pl)

    # Display cumulative returns with trading costs in the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Cumulative_Return'], label='Cumulative Returns', color='green')
    ax.plot(df.index, df['Cumulative_Strategy_Return'], label='Cumulative Strategy Returns with Costs', color='orange')
    ax.set_title('Backtesting Results for Stochastic RSI Strategy with Costs')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    st.pyplot(fig)

    # Display trading metrics with transaction costs and slippage
    st.write("## Trading Metrics with Costs:")
    st.write(f"Total Winning Trades: {total_winning_trades}")
    st.write(f"Total Losing Trades: {total_losing_trades}")
    st.write(f"Total Winning Trade Value: ${total_winning_trade_value:.2f}")
    st.write(f"Total Losing Trade Value: ${total_losing_trade_value:.2f}")
    st.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    st.write(f"Profit Factor: {profit_factor:.2f}")
    st.write(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}")

    st.write("## Total Value of Stock:")
    st.write(f"Total Value of Stock: ${df['Profits_Losses'].iloc[-1]:,.2f}")
    st.write(f"Profit or Loss: ${profit_loss.sum():,.2f}")

    # Display total trading costs
    st.write("## Total Trading Costs:")
    st.write(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
    st.write(f"Total Slippage: ${total_slippage:.2f}")
    st.write(f"Total Trading Costs: ${total_trading_costs:.2f}")

    # Display the number of shares bought or owned
    st.write("## Number of Shares Bought or Owned:")
    st.write(f"Shares Bought: {shares_bought.iloc[0]:.2f}")
    st.write(f"Shares Owned: {df['Shares_Owned'].iloc[-1]:.2f}")

    # Display remaining capital with trading costs
    remaining_capital_message = f"Remaining Capital with Costs: ${remaining_capital:.2f}"
    st.write(remaining_capital_message)
    st.success("Backtesting completed successfully!")

st.write(f"### Calculate Stochastic Relative Strength Index of {ticker_symbol}")
# User input for Stochastic RSI parameters
rsi_window = st.number_input("Enter RSI Window (e.g., 14):", value=14, step=1)
stoch_window = st.number_input("Enter Stochastic Window (e.g., 14):", value=14, step=1)
overbought_threshold = st.number_input("Enter Overbought Threshold (e.g., 0.8):", value=0.8, step=0.01)
oversold_threshold = st.number_input("Enter Oversold Threshold (e.g., 0.2):", value=0.2, step=0.01)

# Example usage
backtest_stochastic_rsi_strategy(tickerDf, rsi_window, stoch_window, overbought_threshold, oversold_threshold, initial_capital, transaction_cost_percentage=0.01, slippage_percentage=0.005)

st.sidebar.title('Options')

option = st.sidebar.selectbox('Which Dashboard?', ('Fundamental Analysis', 'News', 'Stock Predictions', 'Wikipedia', 'Reddit'))

st.header(option)

if option == 'Fundamental Analysis':

    # Create a Ticker object for the user-provided ticker symbol
    stock = yf.Ticker(ticker_symbol)
    stock_info = stock.info
    # Fetch stock information
    ticker_info = yf.Ticker(ticker_symbol).info

    # Get financial data
    balance_sheet = stock.balance_sheet
    quarterly_balance_sheet = stock.quarterly_balance_sheet
    cashflow = stock.cashflow
    financials = stock.financials
    institutional_holders = stock.institutional_holders
    major_holders = stock.major_holders
    mutualfund_holders = stock.mutualfund_holders
    recommendations = stock.recommendations
    splits = stock.splits

    # Retrieve historical data for calculating capital gains (limit to last year)
    history = stock.history(period="1y")

    # Calculate capital gains using a simple example (you would need more inputs for a complete calculation)
    purchase_price = history.iloc[0]['Close']  # Example: purchase price is the first closing price
    sale_price = history.iloc[-1]['Close']  # Example: sale price is the latest closing price

    capital_gains = sale_price - purchase_price

    # Display company information using Streamlit
    st.subheader("Company Information:")
    st.write("Name:", stock_info.get('longName', 'N/A'))  # Using get() to handle missing key
    st.write("Symbol:", stock_info.get('symbol', 'N/A'))
    st.write("City:", stock_info.get('city ', 'N/A'))
    st.write("Website:", stock_info.get('website', 'N/A'))
    st.write("Sector:", stock_info.get('sectorDisp', 'N/A'))
    st.write("Industry:", stock_info.get('industryDisp', 'N/A'))
    st.write("Current Price: $", stock_info.get('currentPrice', 'N/A'))
    st.write("Market Cap: $", stock_info.get('marketCap', 'N/A'))
    st.write("Number of Employees:", stock_info.get('fullTimeEmployees', 'N/A'))
    st.write("Earnings Per Share (EPS):", stock_info.get('trailingEps', 'N/A'))
    st.write("Forward Earnings Per Share (EPS):", stock_info.get('forwardEps', 'N/A'))
    st.write("Dividend Yield:", f"{stock_info.get('trailingAnnualDividendYield', 'N/A') * 100}%")
    st.write("Total Revenue: $", stock_info.get('totalRevenue', 'N/A'))
    st.write("Net Income: $", stock_info.get('netIncomeToCommon', 'N/A'))

    for key, value in ticker_info.items():
        st.write(f"**{key}:** {value}")

    # Display the information you previously fetched
    st.subheader("Company Data:")
    st.write("Balance Sheet:")
    st.write(balance_sheet)

    st.write("Quarterly Balance Sheet:")
    st.write(quarterly_balance_sheet)

    st.write("Cashflow:")
    st.write(cashflow)

    st.write("Financials:")
    st.write(financials)

    st.subheader("Holder Information:")
    st.write("Institutional Holders:")
    st.write(stock.institutional_holders)

    st.write("Major Holders:")
    st.write(stock.major_holders)

    st.write("Mutual Fund Holders:")
    st.write(stock.mutualfund_holders)

    st.write("Recommendations:")
    st.write(recommendations)

    st.write("Splits:")
    st.write(splits)

    # Function to scrape headlines from a given URL
    def scrape_headlines(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        for headline in soup.find_all('h2', class_='headline'):
            headlines.append(headline.text.strip())
        return headlines

    # Function to categorize headlines
    def categorize_headlines(headlines, company_name):
        global_economy_keywords = ['global economy', 'world economy', 'global market', 'economic outlook']
        national_economy_keywords = ['national economy', 'domestic economy', 'economic policy', 'economic indicators']
        industry_keywords = ['industry', 'sector', 'business sector', 'industry trends']
        competitors_keywords = ['competitors', 'rivals', 'industry peers', 'market competition']
    
        categorized_headlines = {
            'Global Economy': [],
            'National Economy': [],
            'Industry': [],
            'Core Competitors': [],
            'Company': []
        }
    
        company_name = str(company_name).lower()
    
        for headline in headlines:
            headline_lower = headline.lower()
            if any(keyword in headline_lower for keyword in global_economy_keywords):
                categorized_headlines['Global Economy'].append(headline)
            if any(keyword in headline_lower for keyword in national_economy_keywords):
                categorized_headlines['National Economy'].append(headline)
            if any(keyword in headline_lower for keyword in industry_keywords):
                categorized_headlines['Industry'].append(headline)
            if any(keyword in headline_lower for keyword in competitors_keywords):
                categorized_headlines['Core Competitors'].append(headline)
            if company_name in headline_lower:
                categorized_headlines['Company'].append(headline)
    
        return categorized_headlines

    # Example usage
    url = 'https://www.bloomberg.com/markets'
    company_name = stock

    headlines = scrape_headlines(url)
    categorized_headlines = categorize_headlines(headlines, company_name)

    # Print categorized headlines
    for category, headlines in categorized_headlines.items():
        st.subheader(f"{category}:")
    for headline in headlines:
        st.subheader(headline)
    st.write('\n')

if option == 'News':
        # Create a Ticker object for the user-provided ticker symbol
        stock = yf.Ticker(ticker_symbol)
        stock_info = stock.info

        # Display stock information
        st.subheader(f"Information for {stock_info.get('longName', ticker_symbol)} ({ticker_symbol})")
        st.write(f"Industry: {stock_info.get('industry', 'N/A')}")
        st.write(f"Sector: {stock_info.get('sector', 'N/A')}")
        st.write(f"Market Cap: {stock_info.get('marketCap', 'N/A')}")

        # News API setup (get your API key from newsapi.org)
        newsapi = NewsApiClient(api_key='d3929aa38779443b9554a474870cf4c5')

        # Fetch news articles
        news_articles = newsapi.get_everything(q=ticker_symbol, language='en', sort_by='relevancy')

        # Add a delay to avoid hitting rate limits
        time.sleep(2)  # Sleep for 2 seconds

       # Display market statistics
        st.subheader("Market Statistics:")
        st.write(f"Opening Price: {tickerDf.iloc[0]['Open']}")
        st.write(f"Previous Close: {tickerDf.iloc[-2]['Close']}")  # Use -2 to get the previous day's close
        st.write(f"Day's High: {tickerDf['High'].max()}")
        st.write(f"Day's Low: {tickerDf['Low'].min()}")
        st.write(f"Trading Volume: {tickerDf.iloc[-1]['Volume']}")

        # Display news articles
        st.subheader("News:")
        news = stock.news
        for i, article in enumerate(news):
            st.header(f"News {i + 1}")
            st.write(f"Title: {article['title']}")
            st.write(f"Publisher: {article['publisher']}")
            st.write(f"Link: {article['link']}")
            st.write(f"Published Time: {article['providerPublishTime']}")
            st.write(f"Type: {article['type']}")
    
            # Check if 'thumbnail' field exists in the article
            if 'thumbnail' in article:
                st.image(article['thumbnail']['resolutions'][0]['url'], width=600, caption='Thumbnail')
                
            else:
                st.write("Thumbnail not available for this article.")
    
            st.write(f"Related Tickers: {', '.join(article['relatedTickers'])}")

        # Fetch news articles
       
        news_articles = newsapi.get_everything(q=ticker_symbol, language='en', sort_by='relevancy')
        
        # Display news articles in a table
        if 'articles' in news_articles:
            articles = news_articles['articles']
            articles_df = pd.DataFrame({
                'Published At': [article['publishedAt'] for article in articles],
                'Title': [article['title'] for article in articles],
                'Description': [article['description'] for article in articles],
                'Source': [article['source']['name'] for article in articles]
            })
            st.subheader("News Articles")
            st.table(articles_df)
        else:
            st.write("No news articles found for this stock.")
        
if option == 'Stock Predictions':
    def load_stock_trend_prediction_dashboard(ticker_symbol, model_file_path='my_model.h5'):
        
        # Fetch historical stock price data based on user input
        try:
            tickerData = yf.Ticker(ticker_symbol)
            tickerDf = tickerData.history(period='1d', start='1989-01-01', end=pd.to_datetime('today'))

            st.subheader(f'Historical Stock Prices of {ticker_symbol}')
            st.write(tickerDf)

            # Display stock price and volume using line charts
            st.subheader(f'Historical Price Chart of {ticker_symbol}')
            st.line_chart(tickerDf.Close)

            st.subheader(f'Historical Volume Trend of {ticker_symbol}')
            st.line_chart(tickerDf.Volume)

        except Exception as e:
            st.error(f"Error fetching data for {ticker_symbol}: {e}")
            return

        df = tickerDf

        st.subheader('Data Head')
        st.write(df.head())

        st.subheader('Describing Data')
        st.write(df.describe())

        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize=(16, 8))
        plt.title('Close Price History')
        plt.plot(df['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        st.pyplot(fig)

        st.subheader('Closing Price vs Time chart with 100MA')
        ma100 = df['Close'].rolling(100).mean()
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100MA')
        plt.plot(df['Close'], label='Close Price')
        plt.legend()
        st.pyplot(fig1)

        st.subheader('Closing Price vs Time chart with 100MA & 30MA')
        ma30 = df['Close'].rolling(30).mean()
        ma100 = df['Close'].rolling(100).mean()
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(ma30, 'r', label='30MA')
        plt.plot(ma100, 'g', label='100MA')
        plt.plot(df['Close'], 'b', label='Close Price')
        plt.legend()
        st.pyplot(fig2)

        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = math.ceil(len(dataset) * 0.8)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        model = load_model(model_file_path)

        test_data = scaled_data[training_data_len - 60:, :]
        x_test = []

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        if x_test.shape[1] < 100:
            pad_width = 100 - x_test.shape[1]
            x_test = np.pad(x_test, ((0, 0), (pad_width, 0), (0, 0)), mode='constant')

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        fig3 = plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD  ($)', fontsize=18)
        plt.plot(train['Close'], label='Train')
        plt.plot(valid[['Close', 'Predictions']], label='Validation & Predictions')
        plt.legend()
        st.pyplot(fig3)

        rmse = np.sqrt(np.mean((predictions - dataset[training_data_len:, :])**2))
        mae = mean_absolute_error(dataset[training_data_len:, :], predictions)
        mse = mean_squared_error(dataset[training_data_len:, :], predictions)
        r2 = r2_score(dataset[training_data_len:, :], predictions)

        st.subheader('Evaluation Metrics')
        st.write('Root Mean Squared Error:', rmse)
        st.write('Mean Absolute Error:', mae)
        st.write('Mean Squared Error:', mse)
        st.write('R2 Score:', r2)
       
        st.subheader('Valid Data with Predictions')
        st.write(valid)

        # Get the last 60 days of closing prices from the original data
        last_60_days = df['Close'].tail(60)

        # Scale the data
        scaled_last_60_days = scaler.transform(last_60_days.values.reshape(-1,1))

        # Reshape the data to match the input shape of your model
        x_test = np.array(scaled_last_60_days)
        x_test = np.reshape(x_test, (1, x_test.shape[0], 1))

        # Pad the data if necessary
        if x_test.shape[1] < 100:
            pad_width = 100 - x_test.shape[1]
            x_test = np.pad(x_test, ((0, 0), (pad_width, 0), (0, 0)), mode='constant')

        # Make the prediction
        prediction = model.predict(x_test)

        # Inverse transform the prediction
        predicted_price_tomorrow = scaler.inverse_transform(prediction)[0][0]

        # Display the predicted price for tomorrow
        st.subheader('Predicted Stock Price Tomorrow')
        st.write(predicted_price_tomorrow)

    load_stock_trend_prediction_dashboard(ticker_symbol, model_file_path='my_model.h5')

if option == 'Wikipedia':
    # Function to fetch Wikipedia text based on search query
    def get_wikipedia_text(query):
        try:
            page = wikipedia.page(query)
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            # If there are multiple options, take the first one
            page = wikipedia.page(e.options[0])
            return page.content
        except wikipedia.exceptions.PageError:
            return "No Wikipedia page found."

    # Create a Ticker object
    stock = yf.Ticker(ticker_symbol)
    stock_info = stock.info
    
    # Fetch Wikipedia text
    wikipedia_text = get_wikipedia_text(stock_info.get('longName', ticker_symbol))
    st.subheader(f"Wikipedia Text of {ticker_symbol}")
    st.write(wikipedia_text)

    # Display stock information
    st.subheader(f"Information for {stock_info.get('longName', ticker_symbol)} ({ticker_symbol})")
    st.write(f"Industry: {stock_info.get('industry', 'N/A')}")
    st.write(f"Sector: {stock_info.get('sector', 'N/A')}")
    st.write(f"Market Cap: {stock_info.get('marketCap', 'N/A')}")


if option == 'Reddit':
    st.subheader('Reddit dashboard logic')