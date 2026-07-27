import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_stock_data(stock_name, interval="1h"):
    ticker = stock_name.upper() + ".NS"
    
    # Interval ke hisaab se period set karo
    if interval == "1h":
        period = "60d"
    else:
        period = "1y"
    
    df = yf.download(ticker, period=period, interval=interval)
    
    if df.empty:
        df = yf.download(stock_name.upper(), period=period, interval=interval)
    
    return df

def prepare_data(df, interval="1h"):
    data = df["Close"].values.reshape(-1, 1)
    
    if data.ndim > 2:
        data = data.squeeze()
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    # Interval ke hisaab se lookback set karo
    if interval == "1h":
        lookback = 24  # 24 ghante
    else:
        lookback = 60  # 60 din
    
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler, lookback

def build_model(lookback):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def predict_price(stock_name, steps=2, interval="1h"):
    try:
        df = get_stock_data(stock_name, interval)
        
        if df.empty:
            print("Data empty!")
            return None, None, None
        
        X, y, scaler, lookback = prepare_data(df, interval)
        
        model = build_model(lookback)
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        
        close_data = df["Close"].values.reshape(-1, 1)
        if close_data.shape[1] > 1:
            close_data = close_data[:, 0].reshape(-1, 1)
        
        scaled_data = scaler.transform(close_data)
        last_sequence = scaled_data[-lookback:].flatten().tolist()
        
        predictions = []
        for _ in range(steps):
            input_data = np.array(last_sequence[-lookback:]).reshape(1, lookback, 1)
            pred = model.predict(input_data, verbose=0)
            pred_value = float(pred.flatten()[0])
            predictions.append(pred_value)
            last_sequence.append(pred_value)
        
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
