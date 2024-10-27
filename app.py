# import streamlit as st
# import subprocess
# from datetime import date
# import time
# import signal

# st.title('🤖 - iTrader')
# animated_txt = 'Intelligent Trading & Financial Advisor Bot'
# text_placeholder = st.empty()
# for i in range(1, len(animated_txt) + 1):
#     text_placeholder.subheader(animated_txt[:i])
#     time.sleep(0.1)

# st.sidebar.header('Trading Options')

# symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')
# cash_at_risk = st.sidebar.slider('Cash at Risk (%)', min_value=0.1, max_value=1.0, value=0.2, step=0.05)

# mode = st.sidebar.radio("Select Mode", ("Live Trading", "Strategy-Backtest"))

# def log_message(message):
#     st.write(f"LOG: {message}")

# # Initialize session state for live trading process
# if 'live_trading_process' not in st.session_state:
#     st.session_state['live_trading_process'] = None

# # Live Trading Execution
# if mode == "Live Trading":
#     if st.sidebar.button('Start Live Trading'):
#         log_message(f"Starting live trading with symbol {symbol} and cash at risk {cash_at_risk}...")
#         try:
#             process = subprocess.Popen(['python', 'run_trader.py', symbol, str(cash_at_risk)])
#             st.session_state['live_trading_process'] = process  # Store the process
#             log_message("Live trading started in a separate process.")
#         except Exception as e:
#             log_message(f"Error during live trading: {e}")

#     if st.sidebar.button('Stop Live Trading'):
#         if st.session_state['live_trading_process'] is not None:
#             st.session_state['live_trading_process'].terminate()  # Terminate the process
#             st.session_state['live_trading_process'].wait()  # Wait for the process to end
#             st.session_state['live_trading_process'] = None  # Clear the stored process
#             log_message("Live trading stopped.")
#         else:
#             log_message("No live trading process is running.")

# # Backtesting Execution
# if mode == "Strategy-Backtest":
#     backtest_start_date = st.sidebar.date_input('Backtest Start Date', date(2022, 10, 1))
#     backtest_end_date = st.sidebar.date_input('Backtest End Date', date(2024, 10, 15))

#     if st.sidebar.button('Start Backtesting'):
#         log_message(f"Starting backtesting from {backtest_start_date} to {backtest_end_date} with symbol {symbol}...")
#         try:
#             subprocess.Popen([
#                 'python', 'run_backtest.py', symbol, str(cash_at_risk),
#                 str(backtest_start_date), str(backtest_end_date)
#             ])
#             log_message("Backtesting started in a separate process.")
#         except Exception as e:
#             log_message(f"Error during backtesting: {e}")


import streamlit as st
import subprocess
from datetime import date
import time
import signal

st.title('🤖 - iTrader')
animated_txt = 'Intelligent Trading & Financial Advisor Bot'
text_placeholder = st.empty()
for i in range(1, len(animated_txt) + 1):
    text_placeholder.subheader(animated_txt[:i])
    time.sleep(0.1)

st.sidebar.header('Trading Options')

symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')
cash_at_risk = st.sidebar.slider('Cash at Risk (%)', min_value=0.1, max_value=1.0, value=0.2, step=0.05)

mode = st.sidebar.radio("Select Mode", ("Live Trading", "Strategy-Backtest"))

# Initialize session state for logging messages and live trading process
if 'log_messages' not in st.session_state:
    st.session_state['log_messages'] = []
if 'live_trading_process' not in st.session_state:
    st.session_state['live_trading_process'] = None

def log_message(message):
    st.session_state.log_messages.append(f"LOG: {message}")  # Append message to session state

# Live Trading Execution
if mode == "Live Trading":
    if st.sidebar.button('Start Live Trading'):
        log_message(f"Starting live trading with symbol {symbol} and cash at risk {cash_at_risk}...")
        try:
            process = subprocess.Popen(['python', 'run_trader.py', symbol, str(cash_at_risk)])
            st.session_state['live_trading_process'] = process  # Store the process
            log_message("Live trading started in a separate process.")
        except Exception as e:
            log_message(f"Error during live trading: {e}")

    if st.sidebar.button('Stop Live Trading'):
        if st.session_state['live_trading_process'] is not None:
            st.session_state['live_trading_process'].terminate()  # Terminate the process
            st.session_state['live_trading_process'].wait()  # Wait for the process to end
            st.session_state['live_trading_process'] = None  # Clear the stored process
            log_message("Live trading stopped.")
        else:
            log_message("No live trading process is running.")

# Backtesting Execution
if mode == "Strategy-Backtest":
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

# Display log messages
# st.subheader("Log Messages")
# for message in st.session_state.log_messages:
#     st.write(message)
