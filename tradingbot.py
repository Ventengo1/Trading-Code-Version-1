import yfinance as yf
import pandas as pd
from datetime import datetime

# Debugging message
print("Hi! The S&P 500 Index trading script with EMA and RSI has started successfully.", flush=True)

# Strategy Parameters
SHORT_EMA_WINDOW = 13  # Short EMA window (13 days)
LONG_EMA_WINDOW = 34   # Long EMA window (34 days)
RSI_WINDOW = 5         # RSI window (5 days)
RSI_OVERBOUGHT = 70    # RSI overbought threshold
RSI_OVERSOLD = 30      # RSI oversold threshold
SYMBOL = "^GSPC"       # S&P 500 Index symbol on Yahoo Finance

def get_historical_data(symbol, days=1500):
    """
    Fetch the last `days` of historical data for the given symbol.
    """
    print(f"Fetching the last {days} days of data for {symbol}...", flush=True)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        print("Error: No historical data fetched. Check your internet connection or symbol.", flush=True)
        return None
    return data

def calculate_ema_and_rsi(data):
    """
    Add short and long exponential moving averages and RSI (with Wilder's Smoothing) to the data.
    """
    # Calculate EMAs
    data["Short_EMA"] = data["Close"].ewm(span=SHORT_EMA_WINDOW, adjust=False).mean()
    data["Long_EMA"] = data["Close"].ewm(span=LONG_EMA_WINDOW, adjust=False).mean()

    # Calculate RSI with Wilder's Smoothing
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Apply Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data

def detect_rsi_divergence(data):
    """
    Detect RSI divergence: Positive or Negative.
    """
    divergence = None

    # Ensure there are at least two rows of data to detect divergence
    if len(data) < 2:
        return divergence

    # Check if the last two rows show divergence
    price_trend = data["Close"].iloc[-2:]  # Last two prices
    rsi_trend = data["RSI"].iloc[-2:]      # Last two RSI values

    # Positive divergence: price makes lower lows, RSI makes higher lows
    if price_trend.iloc[1] < price_trend.iloc[0] and rsi_trend.iloc[1] > rsi_trend.iloc[0]:
        divergence = "Positive Divergence"
    # Negative divergence: price makes higher highs, RSI makes lower highs
    elif price_trend.iloc[1] > price_trend.iloc[0] and rsi_trend.iloc[1] < rsi_trend.iloc[0]:
        divergence = "Negative Divergence"

    return divergence

def get_current_price(symbol):
    """
    Fetch the current price of the given symbol.
    """
    print("Fetching the current S&P 500 Index price...", flush=True)
    ticker = yf.Ticker(symbol)
    price = ticker.history(period="1d")["Close"].iloc[-1]
    print(f"Current S&P 500 Index Price: ${price:.2f}", flush=True)
    return price

def make_decision(data, current_price):
    """
    Make a buy/hold/sell decision based on EMA, RSI, and current price.
    """
    latest_short_ema = data["Short_EMA"].iloc[-1]
    latest_long_ema = data["Long_EMA"].iloc[-1]
    latest_rsi = data["RSI"].iloc[-1]

    print(f"Latest Short EMA (13 days): ${latest_short_ema:.2f}", flush=True)
    print(f"Latest Long EMA (34 days): ${latest_long_ema:.2f}", flush=True)
    print(f"Latest RSI (5 days): {latest_rsi:.2f}", flush=True)

    # Check for divergence
    divergence = detect_rsi_divergence(data)
    if divergence:
        print(f"RSI Divergence Detected: {divergence}", flush=True)

    # Trading decision logic
    if latest_short_ema > latest_long_ema and current_price > latest_short_ema and latest_rsi < RSI_OVERBOUGHT:
        return "Buy"
    elif latest_short_ema < latest_long_ema and current_price < latest_long_ema and latest_rsi > RSI_OVERSOLD:
        return "Sell"
    else:
        return "Hold"

if __name__ == "__main__":
    try:
        print("Starting the S&P 500 Index trading decision bot with EMA, RSI, and divergence detection...", flush=True)
        
        # Step 1: Fetch historical data
        data = get_historical_data(SYMBOL)
        if data is None:
            print("Error: Unable to fetch historical data. Exiting script.", flush=True)
            exit()

        # Step 2: Calculate exponential moving averages and RSI
        data = calculate_ema_and_rsi(data)

        # Step 3: Fetch the current S&P 500 Index price
        current_price = get_current_price(SYMBOL)

        # Step 4: Make a trading decision
        decision = make_decision(data, current_price)
        print(f"Trading Decision: {decision}", flush=True)

        # Final confirmation message
        print("The S&P 500 Index trading decision script with EMA, RSI, and divergence detection has completed successfully.", flush=True)

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
