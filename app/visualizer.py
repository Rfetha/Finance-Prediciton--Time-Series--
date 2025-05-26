import matplotlib.pyplot as plt
import os
import datetime
from config import TAHMIN_FOLDER


class Visualizer:
    """Grafik ve görselleştirme sınıfı"""
    
    def __init__(self, forecast_days=14):
        self.forecast_days = forecast_days
        self._ensure_folder_exists()
    
    def _ensure_folder_exists(self):
        """Tahmin klasörünün varlığını kontrol et"""
        if not os.path.exists(TAHMIN_FOLDER):
            os.makedirs(TAHMIN_FOLDER)
    
    def plot_forecast_chart(self, all_predictions):
        """Tahmin grafiğini çiz"""
        plt.figure(figsize=(15, 8))
        for ticker, data in all_predictions.items():
            plt.plot(data['dates'], data['future_prices'], label=f"{ticker}")
        plt.title(f'Gelecek {self.forecast_days} Gün Fiyat Tahminleri')
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig("tahmin_grafigi.png")
        plt.close()
        
    def plot_normalized_chart(self, all_predictions):
        """Normalize edilmiş tahmin grafiğini çiz"""
        plt.figure(figsize=(15, 8))
        for ticker, data in all_predictions.items():
            # Normalize prices to start at 100
            norm_prices = [price / data['current_price'] * 100 for price in data['future_prices']]
            plt.plot(data['dates'], norm_prices, label=f"{ticker}")
        plt.title('Normalize Edilmiş Fiyat Tahminleri (Başlangıç=100)')
        plt.xlabel('Tarih')
        plt.ylabel('Normalize Edilmiş Fiyat')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.3)  # Reference line at 100
        plt.savefig("normalize_fiyat_grafigi.png")
        plt.close()