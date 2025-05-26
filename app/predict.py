import numpy as np
import pandas as pd


class PricePredictor:
    """Fiyat tahmin sınıfı"""
    
    def __init__(self, forecast_days=14):
        self.forecast_days = forecast_days
    
    def predict_future_prices(self, model, scaler, last_sequence, days=None):
        """Geleceği tahmin et (dinamik RSI & Bollinger ile)"""
        if days is None:
            days = self.forecast_days
            
        sequence = last_sequence.copy()
        future_prices = []
        
        for _ in range(days):
            pred_scaled = model.predict(sequence[np.newaxis, :, :], verbose=0)[0, 0]

            # Dummy array'i son sequence'in son satırındaki özelliklerle doldur
            dummy = sequence[-1].copy()  # Son satır, tüm özellikler
            dummy[0] = pred_scaled       # Sadece close fiyatını tahminle güncelle

            pred_price = scaler.inverse_transform(dummy.reshape(1, -1))[0, 0]
            future_prices.append(pred_price)

            # close_hist ve teknik göstergeler hesaplama
            close_hist = scaler.inverse_transform(sequence)[:, 0]
            close_hist = np.append(close_hist[1:], pred_price)

            # RSI
            delta = np.diff(close_hist)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
            rsi = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

            # Bollinger
            window = 20
            if len(close_hist) < window:
                ma20 = np.mean(close_hist)
                std20 = np.std(close_hist)
            else:
                ma20 = np.mean(close_hist[-window:])
                std20 = np.std(close_hist[-window:])
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20

            new_features = np.array([[pred_price, rsi, ma20, upper, lower]])
            new_scaled = scaler.transform(new_features)

            sequence = np.vstack((sequence[1:], new_scaled))

        return future_prices
    
    def create_prediction_data(self, ticker, current_price, future_prices, df, forecast_days):
        """Tahmin verilerini düzenle"""
        # Mevcut teknik göstergeleri al
        current_rsi = df['RSI'].iloc[-1]
        current_ma20 = df['MA20'].iloc[-1]
        current_upper = df['Upper'].iloc[-1]
        current_lower = df['Lower'].iloc[-1]

        return {
            'ticker': ticker,
            'current_price': current_price,
            'future_prices': future_prices,
            'dates': pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days),
            'current_rsi': current_rsi,
            'current_ma20': current_ma20,
            'current_upper': current_upper,
            'current_lower': current_lower,
            'df': df  # Analiz için tüm dataframe'i saklıyoruz
        }
