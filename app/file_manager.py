import os
import datetime
from config import TAHMIN_FOLDER


class FileManager:
    """Dosya kaydetme ve yönetim sınıfı"""
    
    def __init__(self, forecast_days=14):
        self.forecast_days = forecast_days
        self._ensure_folder_exists()
    
    def _ensure_folder_exists(self):
        """Tahmin klasörünün varlığını kontrol et"""
        if not os.path.exists(TAHMIN_FOLDER):
            os.makedirs(TAHMIN_FOLDER)
    
    def save_recommendations_to_csv(self, recommendations):
        """
        Hisse senedi tavsiyelerini CSV formatında kaydeder
        
        Returns:
            str: Oluşturulan dosyanın adı
        """
        # Şu anki tarih ve saati al
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        filename = os.path.join(TAHMIN_FOLDER, f"tahmin_{timestamp}.csv")
        
        # Tavsiyeleri CSV dosyasına kaydet
        with open(filename, 'w', encoding='utf-8') as f:
            # CSV başlık satırı
            f.write("Ticker,Action,DayCount,Sign,Percentage\n")
            
            # Her hisse için bir satır ekle
            for ticker, rec in recommendations.items():
                # Tavsiye
                action = rec['action']
                # Gün sayısı
                days = self.forecast_days
                # İşaret (+ veya -)
                sign = "+" if rec.get('return_30d', 0) >= 0 else "-"
                # Yüzde değişim (mutlak değer, işaretsiz)
                percentage = f"{abs(rec.get('return_30d', 0)):.2f}"
                
                # CSV satırını yaz
                f.write(f"{ticker},{action},{days},{sign},{percentage}\n")
        
        print(f"\nTavsiyeler '{filename}' CSV dosyasına kaydedildi.")
        return filename

    def save_recommendations_to_txt(self, recommendations):
        """Tavsiyeleri zaman damgalı bir text dosyasına kaydet"""
        # Şu anki tarih ve saati al
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        # Dosya adını oluştur
        filename = os.path.join(TAHMIN_FOLDER, f"tahmin_{timestamp}.txt")
        
        # Tavsiyeleri bir text dosyasına kaydet
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== HİSSE SENEDİ TAVSİYE RAPORU ===\n\n")
            f.write(f"Tarih: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Hisse Senedi Tavsiyeleri:\n")
            for ticker, rec in recommendations.items():
                f.write(f"\n{ticker} - {rec['action']}\n")
                
                # Eğer detaylı veriler varsa yazdır
                if 'current_price' in rec:
                    f.write(f"Mevcut Fiyat: ${rec['current_price']:.2f}\n")
                if 'future_price_7d' in rec:
                    f.write(f"7-Gün Tahmini: ${rec['future_price_7d']:.2f} (%{rec['return_7d']:.2f})\n")
                if 'future_price_30d' in rec:
                    f.write(f"{self.forecast_days}-Gün Tahmini: ${rec['future_price_30d']:.2f} (%{rec['return_30d']:.2f})\n")
                
                # Basit tavsiye formatı için
                if 'forecast_price' in rec:
                    f.write(f"Mevcut: ${rec['current_price']:.2f}, Tahmin: ${rec['forecast_price']:.2f} (%{rec['change_pct']:.2f})\n")
                
                # Nedenler varsa yazdır
                if 'reasons' in rec:
                    f.write("Nedenler:\n")
                    for reason in rec['reasons']:
                        f.write(f"  - {reason}\n")
        
        print(f"\nTavsiyeler '{filename}' dosyasına kaydedildi.")
        return filename