# run_trader.py
from tradingbot import iTrader
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
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

def start_trading(symbol, cash_at_risk):
    strategy = iTrader(name='mlstrat', broker=broker, parameters={"symbol": symbol, "cash_at_risk": cash_at_risk})
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1]
    cash_at_risk = float(sys.argv[2])
    start_trading(symbol, cash_at_risk)
