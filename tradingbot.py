import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema




# Strategy Parameters (Adjust as needed)
SHORT_EMA_WINDOW = 13
LONG_EMA_WINDOW = 34
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
TREND_LOOKBACK = 90
VOLATILITY_LOOKBACK = 20
STD_MULTIPLIER = 1.5




def get_historical_data(symbol, days=1500):
    """Fetch historical data."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        print(f"No data for {symbol}. Check the symbol or internet connection.", flush=True)
        return None
    return data




def calculate_ema_rsi_macd(data):
    """Calculate EMA, RSI, and MACD."""
    data["Short_EMA"] = data["Close"].ewm(span=SHORT_EMA_WINDOW, adjust=False).mean()
    data["Long_EMA"] = data["Close"].ewm(span=LONG_EMA_WINDOW, adjust=False).mean()




    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))




    data["MACD_Fast"] = data["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    data["MACD_Slow"] = data["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    data["MACD_Line"] = data["MACD_Fast"] - data["MACD_Slow"]
    data["MACD_Signal"] = data["MACD_Line"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    data["MACD_Hist"] = data["MACD_Line"] - data["MACD_Signal"]




    return data




def get_current_price(symbol):
    """Fetch the current price."""
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="1d")
    if history.empty:
        print(f"No current price for {symbol}.", flush=True)
        return None
    return history["Close"].iloc[-1]




def analyze_trends(data, lookback=TREND_LOOKBACK):
    """Analyzes price trends using linear regression."""
    trends = []
    for i in range(lookback, len(data)):
        window = data['Close'][i - lookback:i]
        x = np.array(range(len(window))).reshape((-1, 1))
        y = window.values
        model = LinearRegression().fit(x, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        trends.append({'date': data.index[i].to_pydatetime(), 'slope': slope, 'intercept': intercept})
    return trends




def calculate_volatility(data, lookback=VOLATILITY_LOOKBACK, multiplier=STD_MULTIPLIER):
    """Calculate volatility using rolling standard deviation and Bollinger Bands."""
    data['Volatility'] = data['Close'].rolling(window=lookback).std()
    data['Middle_Band'] = data['Close'].rolling(window=lookback).mean()  # Calculate the middle band first




    data['Upper_Band'] = data['Middle_Band'] + (multiplier * data['Volatility'])  # Now use the middle band
    data['Lower_Band'] = data['Middle_Band'] - (multiplier * data['Volatility'])  # Now use the middle band




    data = data.drop(columns=['Middle_Band']) #Drop the middle band column




    return data








def find_turning_points(data):
    """Finds potential turning points (local maxima and minima)."""
    max_indices = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    peaks = data.iloc[max_indices]




    min_indices = argrelextrema(data['Close'].values, np.less, order=5)[0]
    troughs = data.iloc[min_indices]




    return peaks, troughs




def make_decision(data, current_price, trends, peaks, troughs):
    """Make a trading decision (incorporating trend, volatility, and turning points)."""
    latest_short_ema = data["Short_EMA"].iloc[-1]
    latest_long_ema = data["Long_EMA"].iloc[-1]
    latest_rsi = data["RSI"].iloc[-1]
    latest_macd_line = data["MACD_Line"].iloc[-1]
    latest_macd_signal = data["MACD_Signal"].iloc[-1]




    current_trend = None
    for trend in reversed(trends):
        if trend['date'] <= data.index[-1].to_pydatetime():
            current_trend = trend
            break




    if current_trend:
        if current_trend['slope'] > 0.001:
            trend_signal = "Uptrend"
        elif current_trend['slope'] < -0.001:
            trend_signal = "Downtrend"
        else:
            trend_signal = "Sideways"
    else:
        trend_signal = "No Trend"




    is_volatile = data['Volatility'].iloc[-1] > np.mean(data['Volatility'])




    near_peak = False
    near_trough = False
    peak_proximity_days = 5
    trough_proximity_days = 5




    for peak in peaks.index:
        time_diff = (data.index[-1].to_pydatetime() - peak.to_pydatetime()).days
        if 0 <= time_diff <= peak_proximity_days:
            near_peak = True
            break




    for trough in troughs.index:
        time_diff = (data.index[-1].to_pydatetime() - trough.to_pydatetime()).days
        if 0 <= time_diff <= trough_proximity_days:
            near_trough = True
            break




    if (
        trend_signal == "Uptrend"
        and current_price > latest_short_ema
        and latest_rsi < RSI_OVERBOUGHT
        and latest_macd_line > latest_macd_signal
        and not is_volatile
        and not near_peak
    ):
        return "Buy"
    elif (
        trend_signal == "Downtrend"
        and current_price < latest_long_ema
        and latest_rsi > RSI_OVERSOLD
        and latest_macd_line < latest_macd_signal
        and not is_volatile
        and not near_trough
    ):
        return "Sell"
    else:
        return "Hold"




if __name__ == "__main__":
    try:
        print("Starting the multi-stock trading decision bot...", flush=True)


        symbols = input("Enter stock symbols separated by commas: ").strip().upper().split(",")


        for symbol in symbols:
            symbol = symbol.strip()
            print(f"\nAnalyzing {symbol}...\n", flush=True)


            data = get_historical_data(symbol) #<--- This line was missing!
            if data is None:
                continue


            data = calculate_ema_rsi_macd(data)


            trends = analyze_trends(data)
            data = calculate_volatility(data)
            peaks, troughs = find_turning_points(data)


            current_price = get_current_price(symbol)
            if current_price is None:
                continue


            decision = make_decision(data, current_price, trends, peaks, troughs)


            print("\nTrends:")
            for trend in trends:
                print(trend)


            print("\nPeaks:")
            if not peaks.empty:
                print(peaks.to_string())
            else:
                print("No peaks found.")


            print("\nTroughs:")
            if not troughs.empty:
                print(troughs.to_string())
            else:
                print("No troughs found.")


            print("\n--- Analysis Summary ---")
            print(f"Symbol: {symbol}")
            print(f"Current Price: {current_price}")


            # EMA, RSI, MACD Analysis
            latest_short_ema = data["Short_EMA"].iloc[-1]
            latest_long_ema = data["Long_EMA"].iloc[-1]
            latest_rsi = data["RSI"].iloc[-1]
            latest_macd_line = data["MACD_Line"].iloc[-1]
            latest_macd_signal = data["MACD_Signal"].iloc[-1]
            latest_macd_hist = data["MACD_Hist"].iloc[-1]


            print(f"Short EMA ({SHORT_EMA_WINDOW} days): {latest_short_ema:.2f}")
            print(f"Long EMA ({LONG_EMA_WINDOW} days): {latest_long_ema:.2f}")
            print(f"RSI ({RSI_WINDOW} days): {latest_rsi:.2f}")
            print(f"MACD Line ({MACD_FAST}, {MACD_SLOW} days): {latest_macd_line:.2f}")
            print(f"MACD Signal ({MACD_SIGNAL} days): {latest_macd_signal:.2f}")
            print(f"MACD Histogram: {latest_macd_hist:.2f}")


            # ... (Rest of your analysis summary code)


            print(f"Trading Decision for {symbol}: {decision}")
            print("-" * 30)


        print("\nThe multi-stock trading decision script has completed successfully.", flush=True)


    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

