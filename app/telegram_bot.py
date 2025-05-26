import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# External imports
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes
)
import feedparser
import yfinance as yf

# Internal imports - make sure these modules are in your app/ directory
from stock_advisor import StockAdvisorBot
from config import DEFAULT_TICKERS, FORECAST_DAYS, TELEGRAM_KEY, OPENAI_KEY


class TelegramStockBot:
    """
    Telegram Stock Advisor Bot - Refactored Version
    Combines stock analysis, news fetching, and AI-powered commentary
    """
    
    def __init__(self, telegram_token: str, openai_api_key: str):
        """Initialize the bot with API keys and configurations"""
        self.telegram_token = telegram_token
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.supported_tickers = DEFAULT_TICKERS
        self.forecast_days = FORECAST_DAYS
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = (
            "📈 *Hisse Senedi Analiz Botuna Hoş Geldin!*\n\n"
            "🔍 Bir hisse kodu gönder (örn: AAPL)\n"
            "📊 Teknik analiz + AI yorumu + güncel haberler\n\n"
            "Yardım için: /help"
        )
        await update.message.reply_text(welcome_message, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command with detailed usage instructions"""
        help_text = (
            "🤖 *Stock Advisor Bot Yardım Menüsü*\n\n"
            "🔹 *Temel Analiz:*\n"
            "`TSLA` → Hisse analizi + AI yorumu + haberler\n\n"
            
            "🔹 *Komutlar:*\n"
            "📊 `/portfolio TSLA AAPL NVDA` → Çoklu analiz\n"
            "🧠 `/soru TSLA neden düşüyor?` → AI soru-cevap\n"
            "💰 `/yatirim TSLA 10000` → Yatırım simülasyonu\n"
            "📈 `/tavsiye 3` → En iyi 3 hisse önerisi\n"
            "⚖️ `/karsilastir TSLA NVDA` → İki hisse karşılaştırması\n\n"
            
            f"🔹 *Desteklenen Hisseler:*\n"
            f"`{', '.join(self.supported_tickers)}`\n\n"
            
            "⚡ Tüm analizler yapay zeka destekli teknik analiz kullanır"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    def _validate_ticker(self, ticker: str) -> bool:
        """Validate if ticker is supported"""
        return ticker.upper() in self.supported_tickers

    def _validate_tickers(self, tickers: List[str]) -> List[str]:
        """Validate and filter supported tickers"""
        return [t.upper() for t in tickers if self._validate_ticker(t)]

    async def _send_ticker_error(self, update: Update, ticker: str):
        """Send error message for unsupported ticker"""
        error_message = (
            f"⚠️ *{ticker}* desteklenmiyor.\n\n"
            f"📋 Desteklenen hisseler:\n"
            f"`{', '.join(self.supported_tickers)}`"
        )
        await update.message.reply_text(error_message, parse_mode='Markdown')

    def fetch_news(self, ticker: str, limit: int = 3) -> List[str]:
        """Fetch latest news for a given ticker from Yahoo Finance RSS"""
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                return ["Bu hisse için güncel haber bulunamadı."]
            
            headlines = []
            for entry in feed.entries[:limit]:
                title = entry.title
                link = entry.link
                headlines.append(f"• [{title}]({link})")
            
            return headlines
        except Exception as e:
            self.logger.error(f"News fetching error for {ticker}: {str(e)}")
            return ["Haber alınırken hata oluştu."]

    def generate_ai_commentary(self, recommendation: Dict) -> str:
        """Generate AI-powered investment commentary"""
        try:
            reasons = "\n".join(recommendation["reasons"])
            prompt = (
                f"Hisse: {recommendation['ticker']}\n"
                f"Tavsiye: {recommendation['action']}\n"
                f"Mevcut fiyat: ${recommendation['current_price']:.2f}\n"
                f"7 günlük tahmin: ${recommendation['future_price_7d']:.2f} "
                f"(%{recommendation['return_7d']:.2f})\n"
                f"{self.forecast_days} günlük tahmin: ${recommendation['future_price_30d']:.2f} "
                f"(%{recommendation['return_30d']:.2f})\n"
                f"Teknik nedenler:\n{reasons}\n\n"
                f"Yukarıdaki verileri kullanarak yatırımcıya profesyonel, "
                f"anlaşılır ve özet bir yorum yap. 2-3 cümle yeterli."
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen deneyimli bir yatırım danışmanısın. Teknik analiz sonuçlarını sade dille açıklarsın."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"AI commentary error: {str(e)}")
            return "AI yorumu şu anda kullanılamıyor."

    def generate_ai_answer(self, user_question: str) -> str:
        """Generate AI answer based on user question and current news"""
        try:
            # Extract potential tickers from question
            words = user_question.upper().split()
            detected_tickers = [w for w in words if w in self.supported_tickers]
            
            news_context = ""
            if detected_tickers:
                ticker = detected_tickers[0]
                headlines = self.fetch_news(ticker, limit=5)
                news_context = f"\n{ticker} son haberler:\n" + "\n".join(headlines)
            
            system_prompt = (
                "Sen deneyimli bir finansal analiz uzmanısın. Kullanıcının sorularını "
                "güncel haber başlıklarını da dikkate alarak yanıtlarsın. "
                "Cevapların kısa, öz ve yatırımcı odaklı olsun."
            )
            
            user_prompt = (
                f"Soru: {user_question}\n"
                f"Güncel bağlam: {news_context}\n\n"
                f"Yukarıdaki bilgilere dayanarak soruya profesyonel bir yanıt ver."
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"AI answer error: {str(e)}")
            return "Üzgünüm, şu anda bu soruyu yanıtlayamıyorum."

    async def analyze_single_stock(self, update: Update, ticker: str):
        """Analyze a single stock and send comprehensive report"""
        if not self._validate_ticker(ticker):
            await self._send_ticker_error(update, ticker)
            return

        await update.message.reply_text(f"⏳ {ticker} analiz ediliyor...")

        try:
            # Initialize stock advisor and run analysis
            advisor = StockAdvisorBot(tickers=[ticker], forecast_days=self.forecast_days)
            prediction_data = advisor.train_and_predict(ticker)
            recommendation = advisor.recommendation_engine.generate_recommendation(prediction_data)

            # Generate AI commentary
            ai_comment = self.generate_ai_commentary(recommendation)

            # Fetch latest news
            news_headlines = self.fetch_news(ticker)

            # Format response
            response = (
                f"📈 *{ticker} Analiz Raporu*\n\n"
                f"🎯 *Tavsiye:* *{recommendation['action']}*\n"
                f"💵 *Mevcut:* ${recommendation['current_price']:.2f}\n"
                f"📅 *7 Gün:* ${recommendation['future_price_7d']:.2f} "
                f"(%{recommendation['return_7d']:.2f})\n"
                f"📅 *{self.forecast_days} Gün:* ${recommendation['future_price_30d']:.2f} "
                f"(%{recommendation['return_30d']:.2f})\n\n"
                
                f"🔍 *Teknik Nedenler:*\n" + 
                "\n".join([f"• {reason}" for reason in recommendation['reasons']]) + "\n\n"
                
                f"🧠 *AI Yorumu:*\n_{ai_comment}_\n\n"
                
                f"📰 *Son Haberler:*\n" + "\n".join(news_headlines)
            )

            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            self.logger.error(f"Single stock analysis error for {ticker}: {str(e)}")
            await update.message.reply_text(f"❌ {ticker} analiz edilirken hata oluştu: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages (stock tickers)"""
        ticker = update.message.text.strip().upper()
        await self.analyze_single_stock(update, ticker)

    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command for multiple stock analysis"""
        if not context.args:
            await update.message.reply_text(
                "📊 *Portföy Analizi*\n\n"
                f"Kullanım: `/portfolio TSLA AAPL MSFT`\n\n"
                f"Desteklenen: `{', '.join(self.supported_tickers)}`",
                parse_mode='Markdown'
            )
            return

        valid_tickers = self._validate_tickers(context.args)
        if not valid_tickers:
            await update.message.reply_text(
                f"⚠️ Geçerli hisse kodu bulunamadı.\n"
                f"Desteklenen: `{', '.join(self.supported_tickers)}`",
                parse_mode='Markdown'
            )
            return

        await update.message.reply_text(
            f"📊 {', '.join(valid_tickers)} portföy analizi başlatılıyor..."
        )

        try:
            advisor = StockAdvisorBot(tickers=valid_tickers, forecast_days=self.forecast_days)
            
            summary = ""
            for ticker in valid_tickers:
                try:
                    prediction_data = advisor.train_and_predict(ticker)
                    recommendation = advisor.recommendation_engine.generate_recommendation(prediction_data)
                    ai_comment = self.generate_ai_commentary(recommendation)

                    summary += (
                        f"\n\n📈 *{ticker}* → *{recommendation['action']}*\n"
                        f"💵 ${recommendation['current_price']:.2f} → "
                        f"${recommendation['future_price_30d']:.2f} "
                        f"(%{recommendation['return_30d']:.2f})\n"
                        f"🧠 _{ai_comment}_"
                    )
                except Exception as e:
                    summary += f"\n\n❌ {ticker}: Analiz hatası"
                    self.logger.error(f"Portfolio analysis error for {ticker}: {str(e)}")

            await update.message.reply_text(
                f"📊 *Portföy Analiz Raporu*{summary}",
                parse_mode='Markdown'
            )

        except Exception as e:
            self.logger.error(f"Portfolio command error: {str(e)}")
            await update.message.reply_text(f"❌ Portföy analizi sırasında hata: {str(e)}")

    async def question_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /soru command for AI-powered Q&A"""
        if not context.args:
            await update.message.reply_text(
                "❓ *AI Soru-Cevap*\n\n"
                "Kullanım: `/soru TSLA neden düşüyor?`\n"
                "AI, güncel haberleri analiz ederek cevap verir."
            )
            return

        user_question = " ".join(context.args)
        await update.message.reply_text("🧠 Soru analiz ediliyor...")

        try:
            answer = self.generate_ai_answer(user_question)
            await update.message.reply_text(
                f"🤖 *AI Cevabı:*\n{answer}",
                parse_mode='Markdown'
            )
        except Exception as e:
            self.logger.error(f"Question command error: {str(e)}")
            await update.message.reply_text(f"❌ Soru yanıtlanırken hata: {str(e)}")

    async def investment_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /yatirim command for investment simulation"""
        if len(context.args) != 2:
            await update.message.reply_text(
                "💰 *Yatırım Simülasyonu*\n\n"
                "Kullanım: `/yatirim TSLA 10000`\n"
                f"(10000$ TSLA yatırımı {self.forecast_days} gün sonucu)",
                parse_mode='Markdown'
            )
            return

        ticker = context.args[0].upper()
        try:
            amount = float(context.args[1])
        except ValueError:
            await update.message.reply_text("❌ Yatırım miktarı geçerli bir sayı olmalı.")
            return

        if not self._validate_ticker(ticker):
            await self._send_ticker_error(update, ticker)
            return

        await update.message.reply_text(
            f"💰 {ticker} için ${amount:,.2f} yatırım simülasyonu hazırlanıyor..."
        )

        try:
            advisor = StockAdvisorBot(tickers=[ticker], forecast_days=self.forecast_days)
            prediction_data = advisor.train_and_predict(ticker)

            current_price = float(prediction_data['current_price'])
            future_prices = [float(p) for p in prediction_data['future_prices']]
            final_price = future_prices[-1]
            max_price = max(future_prices)
            min_price = min(future_prices)

            # Calculate investment scenarios
            shares = amount / current_price
            final_value = shares * final_price
            max_value = shares * max_price
            min_value = shares * min_price

            final_return = (final_value / amount - 1) * 100
            max_return = (max_value / amount - 1) * 100
            min_return = (min_value / amount - 1) * 100

            # Find best day
            max_index = future_prices.index(max_price)
            best_date = prediction_data['dates'][max_index].strftime("%d %B %Y")

            response = (
                f"💰 *{ticker} Yatırım Simülasyonu*\n\n"
                f"🔹 Yatırım: ${amount:,.2f}\n"
                f"🔹 Hisse Adedi: {shares:.4f}\n\n"
                
                f"📈 *{self.forecast_days} Gün Sonucu:*\n"
                f"💵 Değer: ${final_value:,.2f} (%{final_return:+.2f})\n"
                f"💰 Kar/Zarar: ${final_value-amount:+,.2f}\n\n"
                
                f"🎯 *En İyi Senaryo:*\n"
                f"📅 {best_date}\n"
                f"💵 ${max_value:,.2f} (%{max_return:+.2f})\n\n"
                
                f"⚠️ *En Kötü Senaryo:*\n"
                f"💵 ${min_value:,.2f} (%{min_return:+.2f})"
            )

            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            self.logger.error(f"Investment simulation error for {ticker}: {str(e)}")
            await update.message.reply_text(f"❌ Yatırım simülasyonu hatası: {str(e)}")

    async def recommendation_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tavsiye command for top stock recommendations"""
        try:
            top_n = int(context.args[0]) if context.args else 3
            top_n = min(max(top_n, 1), len(self.supported_tickers))  # Limit between 1 and available tickers
        except ValueError:
            await update.message.reply_text(
                "📈 *En İyi Hisse Tavsiyeleri*\n\n"
                "Kullanım: `/tavsiye 3`\n"
                "(En yüksek getiri potansiyeli olan 3 hisse)",
                parse_mode='Markdown'
            )
            return

        await update.message.reply_text(
            f"📊 En kazançlı {top_n} hisse analiz ediliyor..."
        )

        try:
            advisor = StockAdvisorBot(tickers=self.supported_tickers, forecast_days=self.forecast_days)
            
            # Analyze all tickers and collect positive return stocks
            positive_stocks = []
            
            for ticker in self.supported_tickers:
                try:
                    prediction_data = advisor.train_and_predict(ticker)
                    recommendation = advisor.recommendation_engine.generate_recommendation(prediction_data)
                    
                    if recommendation['return_30d'] > 0:  # Only positive returns
                        positive_stocks.append({
                            'ticker': ticker,
                            'return': recommendation['return_30d'],
                            'action': recommendation['action'],
                            'current_price': recommendation['current_price'],
                            'future_price': recommendation['future_price_30d']
                        })
                except Exception as e:
                    self.logger.error(f"Recommendation analysis error for {ticker}: {str(e)}")
                    continue

            if not positive_stocks:
                await update.message.reply_text(
                    "⚠️ Şu anda pozitif getiri beklentisi olan hisse bulunamadı."
                )
                return

            # Sort by return and get top N
            top_stocks = sorted(positive_stocks, key=lambda x: x['return'], reverse=True)[:top_n]

            response = f"📈 *En Kazançlı {len(top_stocks)} Hisse*\n"
            
            for i, stock in enumerate(top_stocks, 1):
                response += (
                    f"\n*{i}. {stock['ticker']}* - {stock['action']}\n"
                    f"💵 ${stock['current_price']:.2f} → ${stock['future_price']:.2f}\n"
                    f"📈 Beklenen: %{stock['return']:+.2f}\n"
                )

            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            self.logger.error(f"Recommendation command error: {str(e)}")
            await update.message.reply_text(f"❌ Tavsiye analizi hatası: {str(e)}")

    async def compare_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /karsilastir command for comparing two stocks"""
        if len(context.args) != 2:
            await update.message.reply_text(
                "⚖️ *Hisse Karşılaştırması*\n\n"
                "Kullanım: `/karsilastir TSLA NVDA`\n"
                "İki hissenin detaylı karşılaştırması",
                parse_mode='Markdown'
            )
            return

        ticker1, ticker2 = context.args[0].upper(), context.args[1].upper()
        
        if not self._validate_ticker(ticker1) or not self._validate_ticker(ticker2):
            invalid_tickers = [t for t in [ticker1, ticker2] if not self._validate_ticker(t)]
            await update.message.reply_text(
                f"⚠️ Geçersiz hisse: {', '.join(invalid_tickers)}\n"
                f"Desteklenen: `{', '.join(self.supported_tickers)}`",
                parse_mode='Markdown'
            )
            return

        await update.message.reply_text(
            f"⚖️ {ticker1} vs {ticker2} karşılaştırılıyor..."
        )

        try:
            advisor = StockAdvisorBot(tickers=[ticker1, ticker2], forecast_days=self.forecast_days)
            
            # Analyze both stocks
            results = {}
            for ticker in [ticker1, ticker2]:
                prediction_data = advisor.train_and_predict(ticker)
                results[ticker] = advisor.recommendation_engine.generate_recommendation(prediction_data)

            rec1, rec2 = results[ticker1], results[ticker2]

            # Determine winner
            if rec1['return_30d'] > rec2['return_30d']:
                winner = f"🏆 *{ticker1}* daha yüksek getiri potansiyeline sahip"
            elif rec2['return_30d'] > rec1['return_30d']:
                winner = f"🏆 *{ticker2}* daha yüksek getiri potansiyeline sahip"
            else:
                winner = "⚖️ Her iki hisse de benzer getiri potansiyeline sahip"

            response = (
                f"⚖️ *{ticker1} vs {ticker2}*\n\n"
                
                f"*{ticker1}* - {rec1['action']}\n"
                f"💵 ${rec1['current_price']:.2f} → ${rec1['future_price_30d']:.2f}\n"
                f"📈 %{rec1['return_30d']:+.2f} beklenti\n\n"
                
                f"*{ticker2}* - {rec2['action']}\n"
                f"💵 ${rec2['current_price']:.2f} → ${rec2['future_price_30d']:.2f}\n"
                f"📈 %{rec2['return_30d']:+.2f} beklenti\n\n"
                
                f"{winner}"
            )

            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            self.logger.error(f"Compare command error: {str(e)}")
            await update.message.reply_text(f"❌ Karşılaştırma hatası: {str(e)}")

    def setup_handlers(self, application):
        """Setup all command and message handlers"""
        # Command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("portfolio", self.portfolio_command))
        application.add_handler(CommandHandler("soru", self.question_command))
        application.add_handler(CommandHandler("yatirim", self.investment_command))
        application.add_handler(CommandHandler("tavsiye", self.recommendation_command))
        application.add_handler(CommandHandler("karsilastir", self.compare_command))
        
        # Message handler for stock tickers
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    def run(self):
        """Start the bot"""
        application = ApplicationBuilder().token(self.telegram_token).build()
        self.setup_handlers(application)
        
        self.logger.info("Telegram Stock Advisor Bot started...")
        print("🤖 Bot başlatıldı ve çalışıyor...")
        
        application.run_polling()


def main():
    """Main function to run the bot"""
    # Configuration - Replace with your actual tokens
    TELEGRAM_TOKEN = TELEGRAM_KEY
    OPENAI_API_KEY = OPENAI_KEY
    
    # Create and run bot
    bot = TelegramStockBot(
        telegram_token=TELEGRAM_TOKEN,
        openai_api_key=OPENAI_API_KEY
    )
    bot.run()


if __name__ == "__main__":
    main()