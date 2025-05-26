from config import (STRONG_BUY_THRESHOLD, BUY_THRESHOLD, LIGHT_BUY_THRESHOLD,
                   STRONG_SELL_THRESHOLD, SELL_THRESHOLD, LIGHT_SELL_THRESHOLD,
                   RSI_OVERBOUGHT, RSI_OVERSOLD, VOLATILITY_THRESHOLD)


class RecommendationEngine:
    """Hisse senedi tavsiye motoru"""
    
    def __init__(self, forecast_days=14):
        self.forecast_days = forecast_days
    
    def generate_recommendation(self, data):
        """Tahmin verilerine göre tavsiye oluştur"""
        # Tahmin verilerini analiz et
        current_price = float(data['current_price'])
        future_price_7d = float(data['future_prices'][6]) if len(data['future_prices']) > 6 else float(data['future_prices'][-1])
        future_price_30d = float(data['future_prices'][-1])  # 30-gün veya son tahmin
        ticker = data['ticker']

        # Teknik göstergeler
        current_rsi = float(data['current_rsi'])
        current_ma20 = float(data['current_ma20'])
        current_upper = float(data['current_upper'])
        current_lower = float(data['current_lower'])

        # Getiriler
        return_7d = (future_price_7d / current_price - 1) * 100
        return_30d = (future_price_30d / current_price - 1) * 100

        # Temel tavsiyeyi oluştur
        action = ""
        reason = []

        # Uzun vadeli trendi değerlendir
        if return_30d > STRONG_BUY_THRESHOLD:
            action = "GÜÇLÜ AL"
            reason.append(f"{self.forecast_days} gün tahmininde %{return_30d:.2f} yükseliş potansiyeli")
        elif return_30d > BUY_THRESHOLD:
            action = "AL"
            reason.append(f"{self.forecast_days} gün tahmininde %{return_30d:.2f} yükseliş beklentisi")
        elif return_30d > LIGHT_BUY_THRESHOLD:
            action = "HAFİF AL"
            reason.append(f"{self.forecast_days} gün tahmininde %{return_30d:.2f} pozitif getiri")
        elif return_30d < STRONG_SELL_THRESHOLD:
            action = "GÜÇLÜ SAT"
            reason.append(f"{self.forecast_days} gün tahmininde %{abs(return_30d):.2f} düşüş riski")
        elif return_30d < SELL_THRESHOLD:
            action = "SAT"
            reason.append(f"{self.forecast_days} gün tahmininde %{abs(return_30d):.2f} düşüş beklentisi")
        elif return_30d < LIGHT_SELL_THRESHOLD:
            action = "HAFİF SAT"
            reason.append(f"{self.forecast_days} gün tahmininde %{abs(return_30d):.2f} negatif getiri")
        else:
            action = "TUT"
            reason.append(f"{self.forecast_days} gün tahmininde %{return_30d:.2f} sınırlı değişim beklentisi")

        # Teknik göstergeleri değerlendir
        if current_rsi > RSI_OVERBOUGHT:
            reason.append(f"RSI ({current_rsi:.2f}) aşırı alım bölgesinde")
        elif current_rsi < RSI_OVERSOLD:
            reason.append(f"RSI ({current_rsi:.2f}) aşırı satım bölgesinde")

        # Bollinger Bantları
        if current_price > current_upper:
            reason.append("Fiyat üst Bollinger bandının üzerinde (aşırı alım)")
        elif current_price < current_lower:
            reason.append("Fiyat alt Bollinger bandının altında (aşırı satım)")

        # Kısa vadeli trendi değerlendir
        if return_7d > 3 and action not in ["GÜÇLÜ AL", "AL"]:
            reason.append(f"7 gün tahmininde %{return_7d:.2f} yükseliş var")
        elif return_7d < -3 and action not in ["GÜÇLÜ SAT", "SAT"]:
            reason.append(f"7 gün tahmininde %{return_7d:.2f} düşüş var")

        # Yakın gelecekte önemli değişimler
        daily_returns = [data['future_prices'][i+1]/data['future_prices'][i]-1 for i in range(len(data['future_prices'])-1)]
        has_volatile_days = any(abs(ret) > VOLATILITY_THRESHOLD for ret in daily_returns[:7])  # %3'den fazla günlük değişim
        if has_volatile_days:
            reason.append("Yakın dönemde oynaklık bekleniyor")

        return {
            'ticker': ticker,
            'action': action,
            'reasons': reason,
            'current_price': current_price,
            'future_price_7d': future_price_7d,
            'future_price_30d': future_price_30d,
            'return_7d': return_7d,
            'return_30d': return_30d
        }
    
    def generate_portfolio_recommendations(self, all_predictions):
        """Tüm hisseler için tavsiye oluştur"""
        recommendations = {}

        # Her hisse için tavsiye oluştur
        for ticker, data in all_predictions.items():
            recommendations[ticker] = self.generate_recommendation(data)

        return recommendations
    
    def generate_simple_recommendations(self, all_predictions, forecast_days):
        """Hata durumunda basit tavsiyeler oluştur"""
        simple_recommendations = {}
        for ticker, data in all_predictions.items():
            current = float(data['current_price'])
            future = float(data['future_prices'][-1])
            change_pct = (future/current - 1) * 100
            
            if change_pct > 5:
                action = "AL"
            elif change_pct < -5:
                action = "SAT"
            else:
                action = "TUT"
            
            simple_recommendations[ticker] = {
                'action': action,
                'current_price': current,
                'forecast_price': future,
                'change_pct': change_pct
            }
        
        return simple_recommendations