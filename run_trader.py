from tradingbot import iTrader
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

ALPACA_CREDS = {
    "API_KEY": os.getenv("API_KEY"),
    "API_SECRET": os.getenv("API_SECRET"),
    "PAPER": True
}

broker = Alpaca(ALPACA_CREDS)

def start_trading(symbol, cash_at_risk):
    try:
        strategy = iTrader(name='mlstrat', broker=broker, parameters={"symbol": symbol, "cash_at_risk": cash_at_risk})
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
    except Exception as e:
        print(f"An error occurred while starting trading: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_trader.py <symbol> <cash_at_risk>")
        sys.exit(1)

    symbol = sys.argv[1]
    try:
        cash_at_risk = float(sys.argv[2])
    except ValueError:
        print("Error: cash_at_risk must be a float.")
        sys.exit(1)

    print(f"Starting trading for {symbol} with cash at risk: {cash_at_risk}")
    start_trading(symbol, cash_at_risk)
