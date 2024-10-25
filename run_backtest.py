# from tradingbot import iTrader
# from lumibot.backtesting import YahooDataBacktesting
# from lumibot.brokers import Alpaca
# from datetime import datetime
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Alpaca credentials
# ALPACA_CREDS = {
#     "API_KEY": os.getenv("API_KEY"),
#     "API_SECRET": os.getenv("API_SECRET"),
#     "PAPER": True
# }

# # Set the symbol and parameters for backtesting
# symbol = "AAPL"
# cash_at_risk = 0.2

# # Create Alpaca broker (this isn't used for backtesting, but for consistency)
# broker = Alpaca(ALPACA_CREDS)

# # Create an instance of the iTrader strategy
# strategy = iTrader(name='iTraderBacktest', broker=broker, parameters={"symbol": symbol, "cash_at_risk": cash_at_risk})

# # Set the backtesting period
# backtesting_start = datetime(2022, 1, 1)
# backtesting_end = datetime(2024, 1, 1)

# # Run the backtest using YahooDataBacktesting
# try:
#     strategy.backtest(
#         YahooDataBacktesting,
#         backtesting_start=backtesting_start,
#         backtesting_end=backtesting_end,
#         parameters={"symbol": symbol, "cash_at_risk": cash_at_risk}
#     )
#     print("Backtesting completed.")
# except Exception as e:
#     print(f"Error during backtesting: {e}")


# run_backtest.py
from tradingbot import iTrader
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ALPACA_CREDS = {
    "API_KEY": os.getenv("API_KEY"),
    "API_SECRET": os.getenv("API_SECRET"),
    "PAPER": True
}

broker = Alpaca(ALPACA_CREDS)

def run_backtest(symbol, cash_at_risk, start_date, end_date):
    # Convert dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Initialize strategy
    strategy = iTrader(name='mlstrat', broker=broker, parameters={"symbol": symbol, "cash_at_risk": cash_at_risk})

    # Run backtest
    results = strategy.backtest(
        YahooDataBacktesting,
        backtesting_start=start_date,
        backtesting_end=end_date,
        parameters={"symbol": symbol, "cash_at_risk": cash_at_risk}
    )
    print("Backtest Results:", results)

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1]
    cash_at_risk = float(sys.argv[2])
    start_date = sys.argv[3]
    end_date = sys.argv[4]
    run_backtest(symbol, cash_at_risk, start_date, end_date)
