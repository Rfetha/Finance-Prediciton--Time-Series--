import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from config import RSI_WINDOW, BOLLINGER_WINDOW, BOLLINGER_STD


class DataProcessor:
    """Veri çekme ve ön işleme sınıfı"""
    
    def __init__(self, history_period='5y'):
        self.history_period = history_period
    
    def calculate_rsi(self, df, window=RSI_WINDOW):
        """RSI teknik göstergesini hesapla"""
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df['RSI'] = rsi.fillna(50)
        return df
    
    def calculate_bollinger_bands(self, df, window=BOLLINGER_WINDOW, num_std=BOLLINGER_STD):
        """Bollinger Bantlarını hesapla"""
        rolling_mean = df['Close'].rolling(window).mean()
        rolling_std = df['Close'].rolling(window).std()
        df['MA20'] = rolling_mean
        df['Upper'] = rolling_mean + (rolling_std * num_std)
        df['Lower'] = rolling_mean - (rolling_std * num_std)
        # Eksik değerlere MA20 ile doldur (ilk günler)
        df.bfill(inplace=True)
        return df
    
    def get_prepared_data(self, ticker):
        """Veri çekme ve ön işleme"""
        df = yf.download(ticker, period=self.history_period)
        df = self.calculate_rsi(df)
        df = self.calculate_bollinger_bands(df)
        # Öznitelikler
        features = ['Close', 'RSI', 'MA20', 'Upper', 'Lower']
        return df[features]
    
    def create_sequences(self, data, seq_length):
        """LSTM için veri dizileri oluştur"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length][0])  # Close fiyatı
        return np.array(X), np.array(y)
    
    def prepare_training_data(self, ticker, sequence_length):
        """Eğitim için veri hazırla"""
        df = self.get_prepared_data(ticker)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)
        X, y = self.create_sequences(scaled, sequence_length)
        return X, y, scaler, df