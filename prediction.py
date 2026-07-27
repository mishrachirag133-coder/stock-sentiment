import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(stock_name, interval="1h"):
    ticker = stock_name.upper() + ".NS"
    
    if interval == "1h":
        period = "60d"
    else:
        period = "1y"
    
    df = yf.download(ticker, period=period, interval=interval)
    
    if df.empty:
        df = yf.download(stock_name.upper(), period=period, interval=interval)
    
    return df

def prepare_data(df, lookback=24):
    close = df["Close"].values.flatten()
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])
    
    return np.array(X), np.array(y), scaler

def predict_price(stock_name, steps=2, interval="1h"):
    try:
        df = get_stock_data(stock_name, interval)
        
        if df.empty:
            return None, None, None
        
        lookback = 24 if interval == "1h" else 60
        X, y, scaler = prepare_data(df, lookback)
        
        # Random Forest Model
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        # Future predict karo
        close = df["Close"].values.flatten()
        scaled = scaler.transform(close.reshape(-1, 1)).flatten()
        last_sequence = list(scaled[-lookback:])
        
        predictions = []
        for _ in range(steps):
            input_data = np.array(last_sequence[-lookback:]).reshape(1, -1)
            pred = float(model.predict(input_data)[0])
            predictions.append(pred)
            last_sequence.append(pred)
        
        predictions_original = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        current_price = float(df["Close"].iloc[-1].values[0])
        predicted_price = float(predictions_original[-1])
        
        if predicted_price > current_price:
            direction = "UP 📈"
        else:
            direction = "DOWN 📉"
        
        return current_price, predicted_price, direction
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
