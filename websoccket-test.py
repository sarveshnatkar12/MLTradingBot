from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
import os
from dotenv import load_dotenv

load_dotenv()

# Replace your API credentials here
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
base_url = "https://paper-api.alpaca.markets"

# Initialize the Alpaca Stream
stream = Stream(api_key, api_secret, base_url=URL(base_url), data_feed='iex')

# Define a sample callback function to handle incoming data
async def on_data(data):
    print("Received data:", data)

# Subscribe to an example Alpaca channel (e.g., trade updates)
stream.subscribe_trade_updates(on_data)

# Start the stream
try:
    stream.run()
    print("WebSocket connection successful.")
except Exception as e:
    print("WebSocket connection failed:", e)
