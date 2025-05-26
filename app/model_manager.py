import os
import pickle
import datetime
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from config import (MODELS_FOLDER, LSTM_UNITS_1, LSTM_UNITS_2, LSTM_UNITS_3, 
                   DROPOUT_RATE, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT)


class ModelManager:
    """Model oluşturma, kaydetme ve yükleme sınıfı"""
    
    def __init__(self, models_folder=MODELS_FOLDER, model_expiry_days=7, sequence_length=90):
        self.models_folder = models_folder
        self.model_expiry_days = model_expiry_days
        self.sequence_length = sequence_length
        self._ensure_model_folder_exists()
    
    def _ensure_model_folder_exists(self):
        """Modellerin kaydedileceği klasörü oluştur"""
        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)
            print(f"{self.models_folder} klasörü oluşturuldu.")
    
    def build_model(self, input_shape):
        """LSTM modeli oluştur"""
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(LSTM_UNITS_1, return_sequences=True))
        model.add(Dropout(DROPOUT_RATE))
        model.add(LSTM(LSTM_UNITS_2, return_sequences=True))
        model.add(Dropout(DROPOUT_RATE))
        model.add(LSTM(LSTM_UNITS_3))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_model(self, model, X, y):
        """Modeli eğit"""
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                 validation_split=VALIDATION_SPLIT, verbose=0)
        return model
    
    def save_model_and_scaler(self, model, scaler, ticker):
        """Modeli ve scaler'ı kaydet"""
        self._ensure_model_folder_exists()
        
        # Model için dosya yolu
        model_path = os.path.join(self.models_folder, f"{ticker}_model.h5")
        # Scaler için dosya yolu
        scaler_path = os.path.join(self.models_folder, f"{ticker}_scaler.pkl")
        # Info dosyası için yol
        info_path = os.path.join(self.models_folder, f"{ticker}_info.pkl")
        
        # Modeli kaydet
        model.save(model_path)
        # Scaler'ı kaydet
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        
        # Tarih bilgisini kaydet
        info = {
            'created_date': datetime.datetime.now(),
            'sequence_length': self.sequence_length,
            'features': ['Close', 'RSI', 'MA20', 'Upper', 'Lower']
        }
        with open(info_path, 'wb') as file:
            pickle.dump(info, file)
        
        print(f"{ticker} modeli ve ilgili verileri başarıyla kaydedildi.")
    
    def load_saved_model(self, ticker):
        """Kaydedilmiş model ve scaler'ı yükle, yoksa veya eskiyse None döndür"""
        model_path = os.path.join(self.models_folder, f"{ticker}_model.h5")
        scaler_path = os.path.join(self.models_folder, f"{ticker}_scaler.pkl")
        info_path = os.path.join(self.models_folder, f"{ticker}_info.pkl")
        
        # Model dosyaları var mı kontrol et
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(info_path)):
            return None, None, None
        
        # Model bilgisini yükle
        with open(info_path, 'rb') as file:
            info = pickle.load(file)
        
        # Model eskimiş mi kontrol et
        if (datetime.datetime.now() - info['created_date']).days > self.model_expiry_days:
            print(f"{ticker} modeli {self.model_expiry_days} günden eski, yeniden eğitilecek.")
            return None, None, None
        
        # Sequence length değişmiş mi kontrol et
        if info['sequence_length'] != self.sequence_length:
            print(f"{ticker} modelinin sequence length'i ({info['sequence_length']}) " 
                f"mevcut ayardan ({self.sequence_length}) farklı, model yeniden eğitilecek.")
            return None, None, None
        
        try:
            # Modeli yükle
            model = load_model(model_path)
            # Scaler'ı yükle
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
            
            print(f"{ticker} modeli başarıyla yüklendi. (Eğitim tarihi: {info['created_date']})")
            return model, scaler, info
        except Exception as e:
            print(f"Model yükleme hatası {ticker}: {str(e)}")
            return None, None, None
    
    def save_last_sequence(self, ticker, last_sequence, df):
        """Son sequence ve DF'nin son 100 satırını kaydet (güncellemeler için)"""
        self._ensure_model_folder_exists()
        sequence_path = os.path.join(self.models_folder, f"{ticker}_last_sequence.pkl")
        df_path = os.path.join(self.models_folder, f"{ticker}_last_df.pkl")
        
        # Son sequence'i kaydet
        with open(sequence_path, 'wb') as file:
            pickle.dump(last_sequence, file)
        
        # DF'nin son 100 satırını kaydet (RAM tasarrufu için)
        with open(df_path, 'wb') as file:
            pickle.dump(df.tail(100), file)
    
    def load_last_sequence(self, ticker):
        """Kaydedilmiş son sequence ve DF'yi yükle"""
        sequence_path = os.path.join(self.models_folder, f"{ticker}_last_sequence.pkl")
        df_path = os.path.join(self.models_folder, f"{ticker}_last_df.pkl")
        
        if not (os.path.exists(sequence_path) and os.path.exists(df_path)):
            return None, None
        
        try:
            # Son sequence'i yükle
            with open(sequence_path, 'rb') as file:
                last_sequence = pickle.load(file)
            
            # DF'yi yükle
            with open(df_path, 'rb') as file:
                df = pickle.load(file)
            
            return last_sequence, df
        except Exception as e:
            print(f"Son sequence yükleme hatası {ticker}: {str(e)}")
            return None, None