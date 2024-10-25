import streamlit as st
import subprocess
from datetime import date

st.title('iTrader Trading Bot')
st.sidebar.header('Trading Options')

symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')
cash_at_risk = st.sidebar.slider('Cash at Risk (%)', min_value=0.1, max_value=1.0, value=0.2, step=0.05)

mode = st.sidebar.radio("Select Mode", ("Live Trading", "Backtesting"))

def log_message(message):
    st.write(f"LOG: {message}")

# Live Trading Execution
if mode == "Live Trading":
    if st.sidebar.button('Start Live Trading'):
        log_message(f"Starting live trading with symbol {symbol} and cash at risk {cash_at_risk}...")
        try:
            subprocess.Popen(['python', 'run_trader.py', symbol, str(cash_at_risk)])
            log_message("Live trading started in a separate process.")
        except Exception as e:
            log_message(f"Error during live trading: {e}")

# Backtesting Execution
if mode == "Backtesting":
    backtest_start_date = st.sidebar.date_input('Backtest Start Date', date(2022, 10, 1))
    backtest_end_date = st.sidebar.date_input('Backtest End Date', date(2024, 10, 15))

    if st.sidebar.button('Start Backtesting'):
        log_message(f"Starting backtesting from {backtest_start_date} to {backtest_end_date} with symbol {symbol}...")
        try:
            subprocess.Popen([
                'python', 'run_backtest.py', symbol, str(cash_at_risk),
                str(backtest_start_date), str(backtest_end_date)
            ])
            log_message("Backtesting started in a separate process.")
        except Exception as e:
            log_message(f"Error during backtesting: {e}")
