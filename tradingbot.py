import logging
from dotenv import load_dotenv
import os
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.backtesting import YahooDataBacktesting
from datetime import datetime
from alpaca_trade_api.rest import REST
from pandas import Timedelta
from finbert_utils import estimate_sentiment

# Set up logging
logging.basicConfig(filename="MLTradingBot/logs/itrader_log.txt", level=logging.INFO)

def log_trade_details(order):
    filled_price = getattr(order, 'filled_avg_price', 'N/A')  # 'N/A' if filled_avg_price is not present
    logging.info(f"Trade executed: {order.symbol}, {order.quantity} shares at {filled_price}, status: {order.status}")

# Load the .env file for API keys
load_dotenv()

# Access the API keys from the .env file
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Alpaca configurations for both live and paper trading
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True  # Change to False for live trading
}

BASE_URL = "https://paper-api.alpaca.markets" if ALPACA_CREDS["PAPER"] else "https://api.alpaca.markets"

class iTrader(Strategy):
    def initialize(self, symbol: str = "AAPL", cash_at_risk: float = .2):
        self.symbol = symbol
        self.sleeptime = "24H"  # Will make run every 24H
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(API_KEY, API_SECRET, BASE_URL)  # Alpaca REST API

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > .7:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * .95
                )
                self.submit_order(order)
                log_trade_details(order)  # Log the buying order
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .7:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * .8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                log_trade_details(order)  # Log the selling order
                self.last_trade = "sell"

# Common broker setup for both live trading and backtesting
broker = Alpaca(ALPACA_CREDS)

