import os
os.chdir("app")

from data_processor import DataProcessor
from model_manager import ModelManager
from predict import PricePredictor
from recomend import RecommendationEngine
from visualizer import Visualizer
from file_manager import FileManager
from config import (DEFAULT_TICKERS, FORECAST_DAYS, HISTORY_PERIOD, 
                   SEQUENCE_LENGTH, MODELS_FOLDER, MODEL_EXPIRY_DAYS,
                   INVESTMENT_AMOUNT)


class StockAdvisorBot:
    def __init__(self, tickers=None, forecast_days=FORECAST_DAYS, 
                 history_period=HISTORY_PERIOD, sequence_length=SEQUENCE_LENGTH, 
                 models_folder=MODELS_FOLDER, model_expiry_days=MODEL_EXPIRY_DAYS, 
                 investment_amount=INVESTMENT_AMOUNT):
        """
        Hisse senedi tavsiye botu sınıfı - Refactored Version
        
        Parameters:
            tickers (list): Analiz edilecek hisse senedi sembolleri
            forecast_days (int): Tahmin edilecek gün sayısı
            history_period (str): Geçmiş veri periyodu (yfinance formatı)
            sequence_length (int): LSTM modeli için kullanılacak dizi uzunluğu
            models_folder (str): Modellerin kaydedileceği klasör
            model_expiry_days (int): Modelin yeniden eğitilmesi için geçmesi gereken gün sayısı
            investment_amount (float): Toplam yatırım miktarı
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.forecast_days = forecast_days
        
        # Bileşenleri başlat
        self.data_processor = DataProcessor(history_period)
        self.model_manager = ModelManager(models_folder, model_expiry_days, sequence_length)
        self.predictor = PricePredictor(forecast_days)
        self.recommendation_engine = RecommendationEngine(forecast_days)
        self.visualizer = Visualizer(forecast_days)
        self.file_manager = FileManager(forecast_days)
        
        # Tahminleri ve tavsiyeleri saklayacak değişkenler
        self.all_predictions = {}
        self.recommendations = {}
        
    def train_and_predict(self, ticker):
        """Ana işlev: Hisse için model oluştur ve tahmin yap"""
        print(f"\n{ticker} için işlem başlıyor...")
        
        # Önce kaydedilmiş modeli kontrol et
        model, scaler, info = self.model_manager.load_saved_model(ticker)
        
        if model is not None and scaler is not None:
            # Kaydedilmiş modeli kullan
            last_sequence, df = self.model_manager.load_last_sequence(ticker)
            
            if last_sequence is not None and df is not None:
                print(f"{ticker} için kaydedilmiş son veri kullanılıyor.")
            else:
                # Son veri yok, yeniden veri çek ama eğitim yapma
                print(f"{ticker} için kaydedilmiş model var ama son veri yok. Veri çekiliyor...")
                df = self.data_processor.get_prepared_data(ticker)
                scaled = scaler.transform(df.values)
                last_sequence = scaled[-self.model_manager.sequence_length:]
                # Son veriyi kaydet
                self.model_manager.save_last_sequence(ticker, last_sequence, df)
        else:
            # Yeni model eğit
            print(f"{ticker} için veri hazırlanıyor ve model eğitiliyor...")
            X, y, scaler, df = self.data_processor.prepare_training_data(ticker, self.model_manager.sequence_length)
            
            # Model oluştur ve eğit
            model = self.model_manager.build_model((X.shape[1], X.shape[2]))
            model = self.model_manager.train_model(model, X, y)
            
            # Modeli kaydet
            self.model_manager.save_model_and_scaler(model, scaler, ticker)
            
            # Son sequence'i al
            scaled = scaler.transform(df.values)
            last_sequence = scaled[-self.model_manager.sequence_length:]
            
            # Son sequence'i kaydet
            self.model_manager.save_last_sequence(ticker, last_sequence, df)
        
        # Gelecek tahminini yap
        future_prices = self.predictor.predict_future_prices(model, scaler, last_sequence)
        
        # Tahmin verilerini düzenle
        return self.predictor.create_prediction_data(
            ticker, df['Close'].iloc[-1], future_prices, df, self.forecast_days
        )
    
    def run(self):
        """Ana işlev: Tüm süreci çalıştır"""
        print("Hisse Senedi Tavsiye Botu başlatılıyor...")
        print(f"Modeller '{self.model_manager.models_folder}' klasöründe saklanacak.")
        
        # Tüm hisseler için tahmin modellerini oluştur
        print("Tüm hisseler için tahmin modellerini oluşturma veya yükleme ve geleceği tahmin etme...")
        for ticker in self.tickers:
            try:
                self.all_predictions[ticker] = self.train_and_predict(ticker)
            except Exception as e:
                print(f"Hata: {ticker} için tahmin yapılamadı - {str(e)}")
        
        # Tüm hisseler için tavsiyeler oluştur
        print("\n\n=== HİSSE SENEDİ TAVSİYE RAPORU ===")
        
        try:
            # Portföy ağırlıkları olmadan tavsiyeler oluştur
            self.recommendations = self.recommendation_engine.generate_portfolio_recommendations(self.all_predictions)
            
            # Tavsiyeleri göster
            print("\nHisse Senedi Tavsiyeleri:")
            for ticker, rec in self.recommendations.items():
                print(f"\n{ticker} - {rec['action']}")
                print(f"Mevcut Fiyat: ${rec['current_price']:.2f}")
                print(f"7-Gün Tahmini: ${rec['future_price_7d']:.2f} (%{rec['return_7d']:.2f})")
                print(f"{self.forecast_days}-Gün Tahmini: ${rec['future_price_30d']:.2f} (%{rec['return_30d']:.2f})")
                print("Nedenler:")
                for reason in rec['reasons']:
                    print(f"  - {reason}")
        
        except Exception as e:
            print(f"Tavsiye oluşturma sırasında hata oluştu: {str(e)}")
            print("Basit tavsiyeler oluşturuluyor...")
            
            # Basit tavsiyeler oluştur
            self.recommendations = self.recommendation_engine.generate_simple_recommendations(
                self.all_predictions, self.forecast_days
            )
            
            # Basit tavsiyeleri göster
            print("\nBasitleştirilmiş Hisse Senedi Tavsiyeleri:")
            for ticker, rec in self.recommendations.items():
                print(f"{ticker}: {rec['action']} - Mevcut: ${rec['current_price']:.2f}, "
                      f"Tahmin: ${rec['forecast_price']:.2f} (%{rec['change_pct']:.2f})")
        
        # Grafikleri çiz
        self.visualizer.plot_forecast_chart(self.all_predictions)
        self.visualizer.plot_normalized_chart(self.all_predictions)
        print("\nİşlem tamamlandı. Grafik dosyaları kaydedildi.")
        
        # Dosyaları kaydet
        if self.recommendations:
            self.file_manager.save_recommendations_to_csv(self.recommendations)
            print("\nİşlem tamamlandı. CSV kaydedildi.")
            self.file_manager.save_recommendations_to_txt(self.recommendations)
            print("\nİşlem tamamlandı. TXT kaydedildi.")
            
        return self.recommendations


# Uygulama çalıştırma
if __name__ == "__main__":
    # Bot oluştur ve çalıştır
    advisor_bot = StockAdvisorBot()
    results = advisor_bot.run()