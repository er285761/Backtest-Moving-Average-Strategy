# Backtest-Moving-Average-Strategy
This code is a Streamlit application that provides functionalities for backtesting a Moving Average Crossover strategy on stock data. The application allows users to input a stock ticker symbol, retrieve historical stock prices from Yahoo Finance, and analyze the stock's performance using a technical trading strategy.
Streamlit App Setup
Libraries Import: The code imports necessary libraries such as streamlit, pandas, numpy, yfinance, matplotlib, and nltk, among others, to handle data processing, visualization, and natural language processing.

Streamlit Title and User Inputs:

Ticker Symbol: User can input the stock ticker symbol (default is 'TSLA').
Start Date: User can input the start date for retrieving historical data (default is '2021-01-01').
Data Retrieval and Display
Fetching Historical Data:
The application uses yfinance to fetch historical stock data for the given ticker symbol and start date.
The fetched data is displayed as a DataFrame and also as line charts showing historical closing prices and volume trends.
Moving Average Crossover Strategy
User Inputs for Strategy Parameters:
Initial Capital: The amount of initial capital for the backtest.
Short-term and Long-term Moving Averages: User inputs for the window sizes of the moving averages.
Backtesting Function
Backtest Function Definition:
backtest_ma_crossover_strategy(df, short_ma_window, long_ma_window, initial_capital, transaction_cost_percentage, slippage_percentage):
Calculate Moving Averages: The short-term and long-term moving averages are calculated.
Position Calculation: Positions (buy/sell/hold) are determined based on the moving averages.
Strategy Returns: Strategy returns are calculated based on daily returns and the position held.
Transaction Costs and Slippage: These costs are incorporated into the backtest to make it realistic.
Performance Metrics: Various performance metrics such as cumulative returns, win/loss ratio, profit factor, and risk/reward ratio are computed.
Visualizations:
A combined chart showing closing prices, moving averages, and buy/sell signals.
Cumulative profits and losses.
Cumulative returns with trading costs.
Trading Metrics: Display of total winning trades, losing trades, total trade values, and remaining capital.
Final Outputs
Displaying Results: The results of the backtest, including charts and performance metrics, are displayed using Streamlit's st.write and st.pyplot functions.
The application provides a comprehensive tool for backtesting a moving average crossover strategy, allowing users to understand how the strategy would have performed historically, including the impact of transaction costs and slippage.
