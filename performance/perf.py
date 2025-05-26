import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import pickle
import os
os.chdir("performance")  

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GRU, SimpleRNN, Conv1D, MaxPooling1D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping



class StockAdvisorBot:
    def __init__(self, tickers, forecast_days=14, history_period='5y',
                 sequence_length=90, models_folder='models', tahmin_folder = "tahmin", 
                 model_expiry_days=7, investment_amount=10000, report_file="stock_analysis_report.txt"):
        
        self.TICKERS = tickers
        self.FORECAST_DAYS = forecast_days
        self.HISTORY_PERIOD = history_period
        self.SEQUENCE_LENGTH = sequence_length
        self.MODELS_FOLDER = models_folder
        self.MODEL_EXPIRY_DAYS = model_expiry_days
        self.INVESTMENT_AMOUNT = investment_amount
        self.TAHMIN_FOLDER = tahmin_folder
        self.REPORT_FILE = report_file
        
        # Tahminleri ve tavsiyeleri saklayacak değişkenler
        self.all_predictions = {}
        self.recommendations = {}
        
        # Ana rapor dosyasını başlat
        self._initialize_report_file()
        
    def _initialize_report_file(self):
        """Ana rapor dosyasını başlat"""
        with open(self.REPORT_FILE, "w", encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HISSE SENEDİ ANALİZ RAPORU\n")
            f.write("=" * 60 + "\n")
            f.write(f"Rapor Tarihi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analiz Edilen Hisseler: {', '.join(self.TICKERS)}\n")
            f.write(f"Tahmin Süresi: {self.FORECAST_DAYS} gün\n")
            f.write(f"Geçmiş Veri Periyodu: {self.HISTORY_PERIOD}\n")
            f.write(f"Sekans Uzunluğu: {self.SEQUENCE_LENGTH}\n")
            f.write("=" * 60 + "\n\n")
    
    def _write_to_report(self, content):
        """Ana rapor dosyasına içerik ekle"""
        with open(self.REPORT_FILE, "a", encoding='utf-8') as f:
            f.write(content + "\n")
    
    def calculate_rsi(self, df, window=14):
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
    
    def calculate_bollinger_bands(self, df, window=20, num_std=3):
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
        df = yf.download(ticker, period=self.HISTORY_PERIOD)
        df = self.calculate_rsi(df)
        df = self.calculate_bollinger_bands(df)
        # Öznitelikler
        features = ['Close', 'RSI', 'MA20', 'Upper', 'Lower']
        return df[features]
    
    def create_sequences(self, data, seq_length=None):
        """LSTM için veri dizileri oluştur"""
        if seq_length is None:
            seq_length = self.SEQUENCE_LENGTH
            
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length][0])  # Close fiyatı
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_gru_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(GRU(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_simple_rnn_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SimpleRNN(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_cnn_lstm_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Reshape((1, -1)))  # LSTM beklerken şekli ayarlamak için
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_dense_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=(input_shape[0]*input_shape[1],)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    
    def train_multiple_models(self, X_train, y_train, X_val, y_val, epochs, batch_size, ticker):
        models = {
            "LSTM": self.build_lstm_model,
            "GRU": self.build_gru_model,
            "SimpleRNN": self.build_simple_rnn_model,
            "CNN_LSTM": self.build_cnn_lstm_model,
            "Dense": self.build_dense_model,
        }

        results = {}
    
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Ticker başlığını rapora ekle
        self._write_to_report(f"\n{'='*40}")
        self._write_to_report(f"{ticker} HİSSESİ ANALİZ SONUÇLARI")
        self._write_to_report(f"{'='*40}")

        for name, builder in models.items():
            print(f"\n--- {name} modeli eğitiliyor ---")
            
            # Model başlığını rapora ekle
            self._write_to_report(f"\n{'-'*25}")
            self._write_to_report(f"{name} MODELİ")
            self._write_to_report(f"{'-'*25}")
            
            # Eğer Dense modeli ise input şekli farklı, onu düzel
            if name == "Dense":
                X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
                X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
                model = builder(input_shape=(X_train.shape[1], X_train.shape[2]))
                history = model.fit(
                    X_train_reshaped, y_train,
                    validation_data=(X_val_reshaped, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=2
                )
            else:
                model = builder(input_shape=X_train.shape[1:])
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],     
                    verbose=2
                )
            
            # Tahmin yap ve metrik hesapla
            if name == "Dense":
                y_pred = model.predict(X_val_reshaped)
            else:
                y_pred = model.predict(X_val)
            
            print(f"\n{name} modeli performans metrikleri:")
            metrics = self.calculate_metrics(y_true=y_val, y_pred=y_pred, 
                                           model_name=name, write_to_report=True)
            results[name] = metrics

        # Özet karşılaştırma tablosunu rapora ekle
        self._write_to_report(f"\n{ticker} MODEL KARŞILAŞTIRMA ÖZETİ:")
        self._write_to_report("-" * 50)
        for model_name, metrics in results.items():
            line = f"{model_name:<12}: RMSE = {metrics['RMSE']:.4f}, R2 = {metrics['R2']:.4f}"
            self._write_to_report(line)
            print(line)  # Terminale de yazdır

        return results

    def evaluate_model(self, model, X_test, y_test):
        """Modelin test/veri setindeki kaybını döner"""
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (MSE): {loss:.6f}")
        return loss

    def calculate_mape(self, y_true, y_pred, epsilon=1e-10):
        """Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    def calculate_smape(self, y_true, y_pred, epsilon=1e-10):
        """Symmetric Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    def calculate_mase(self, y_true, y_pred):
        """
        Mean Absolute Scaled Error
        MASE = MAE / MAE of naive forecast
        naive forecast: y_t = y_{t-1}
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n = y_true.shape[0]
        mae_model = np.mean(np.abs(y_true - y_pred))
        naive_forecast_errors = np.abs(y_true[1:] - y_true[:-1])
        mae_naive = np.mean(naive_forecast_errors)
        if mae_naive == 0:
            return np.nan  # Bölme sıfıra denk gelirse nan döner
        return mae_model / mae_naive

    def calculate_wape(self, y_true, y_pred):
        """Weighted Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    def calculate_metrics(self, y_true, y_pred, model_name=None, write_to_report=False):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = self.calculate_mape(y_true, y_pred)
        smape = self.calculate_smape(y_true, y_pred)
        mase = self.calculate_mase(y_true, y_pred)
        wape = self.calculate_wape(y_true, y_pred)

        # Konsola yazdır
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R2 Score: {r2:.6f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"sMAPE: {smape:.4f}%")
        print(f"MASE: {mase:.6f}")
        print(f"WAPE: {wape:.4f}%")

        # Ana rapora yaz
        if write_to_report:
            self._write_to_report(f"MSE: {mse:.6f}")
            self._write_to_report(f"RMSE: {rmse:.6f}")
            self._write_to_report(f"MAE: {mae:.6f}")
            self._write_to_report(f"R2 Score: {r2:.6f}")
            self._write_to_report(f"MAPE: {mape:.4f}%")
            self._write_to_report(f"sMAPE: {smape:.4f}%")
            self._write_to_report(f"MASE: {mase:.6f}")
            self._write_to_report(f"WAPE: {wape:.4f}%")

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'sMAPE': smape,
            'MASE': mase,
            'WAPE': wape
        }

def main():
    # Kullanılacak hisse senedi ve parametreler
    tickers = ['AAPL']  # Örnek olarak Apple hisse senedi
    forecast_days = 14
    history_period = '5y'
    sequence_length = 90
    epochs = 100
    batch_size = 32

    # Botu oluştur
    bot = StockAdvisorBot(tickers=tickers, forecast_days=forecast_days,
                          history_period=history_period, sequence_length=sequence_length)

    for ticker in tickers:
        print(f"\n=== {ticker} için veri hazırlanıyor ===")
        df = bot.get_prepared_data(ticker)

        # Veriyi ölçeklendir
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        # Diziler oluştur
        X, y = bot.create_sequences(scaled_data, seq_length=sequence_length)

        # Eğitim ve test seti ayır
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Model eğitim ve değerlendirme
        print(f"\n{ticker} için modeller eğitiliyor...")
        results = bot.train_multiple_models(X_train, y_train, X_test, y_test, 
                                          epochs=epochs, batch_size=batch_size, ticker=ticker)
        
    # Rapor tamamlandı mesajı
    bot._write_to_report(f"\n{'='*60}")
    bot._write_to_report("RAPOR TAMAMLANDI")
    bot._write_to_report(f"{'='*60}")
    
    print(f"\nTüm sonuçlar {bot.REPORT_FILE} dosyasına başarıyla kaydedildi.")



if __name__ == "__main__":
    main()