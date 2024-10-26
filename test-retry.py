import logging
from dotenv import load_dotenv
import os
from lumibot.brokers import Alpaca
from alpaca_trade_api.rest import REST
from datetime import datetime, timedelta
from time import sleep
from finbert_utils import estimate_sentiment

# Set up logging
log_directory = "MLTradingBot/logs"
log_file = os.path.join(log_directory, "itrader_log.txt")
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO)

# Load the .env file for API keys
load_dotenv()

# Access the API keys from the .env file
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# Alpaca REST API connection with retries
def connect_with_retry(retry_limit=3, retry_delay=5):
    attempt = 0
    api_instance = None
    while attempt < retry_limit:
        try:
            api_instance = REST(API_KEY, API_SECRET, BASE_URL)
            account = api_instance.get_account()
            logging.info("Connection to Alpaca API established successfully.")
            print("Connection to Alpaca API established successfully.")
            logging.info(f"Account Status: {account.status}")
            return api_instance
        except Exception as e:
            logging.warning(f"Connection attempt {attempt + 1} failed: {e}")
            print(f"Connection attempt {attempt + 1} failed: {e}")
            attempt += 1
            if attempt < retry_limit:
                sleep(retry_delay)
            else:
                logging.error("Max retries reached. Connection could not be established.")
                raise e
    return api_instance

# Function to get dates
def get_dates():
    today = datetime.now()
    three_days_prior = today - timedelta(days=3)
    return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

# Function to get sentiment
def get_sentiment(api, symbol):
    today, three_days_prior = get_dates()
    print(f"Fetching news for symbol: {symbol} from {three_days_prior} to {today}")
    logging.info(f"Fetching news for symbol: {symbol} from {three_days_prior} to {today}")

    # Fetch news from the Alpaca API
    news = api.get_news(symbol, start=three_days_prior, end=today)
    news_headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
    print(f"News headlines fetched: {news_headlines}")
    logging.info(f"News headlines fetched: {news_headlines}")

    # Check if news headlines were retrieved
    if not news_headlines:
        print("No news headlines retrieved from Alpaca API.")
        logging.info("No news headlines retrieved from Alpaca API.")
        return None, None  # No headlines to analyze

    # Perform sentiment analysis
    probability, sentiment = estimate_sentiment(news_headlines)
    print(f"Sentiment analysis result: Probability - {probability}, Sentiment - {sentiment}")
    logging.info(f"Sentiment analysis result: Probability - {probability}, Sentiment - {sentiment}")

    return probability, sentiment

# Main code
if __name__ == "__main__":
    # Connect to Alpaca API with retries
    api = connect_with_retry()

    # Symbol for sentiment check
    symbol = "GOOG"

    # Get sentiment for the symbol
    probability, sentiment = get_sentiment(api, symbol)
    if probability is not None and sentiment is not None:
        print(f"Sentiment Probability: {probability}, Sentiment: {sentiment}")
    else:
        print("Could not fetch sentiment due to lack of news.")
