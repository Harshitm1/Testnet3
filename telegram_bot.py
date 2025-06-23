import requests
from config import TELEGRAM_CONFIG, TRADING_CONFIG
from logger import setup_logger

logger = setup_logger('telegram_bot')

class TelegramBot:
    def __init__(self):
        """Initialize Telegram bot with configuration."""
        self.base_url = f"https://api.telegram.org/bot{TELEGRAM_CONFIG['bot_token']}"
        self.chat_id = TELEGRAM_CONFIG['chat_id']
        print(f"TELEGRAM DEBUG INIT: base_url = {self.base_url} chat_id = {self.chat_id}")

    def send_message(self, message):
        """Send a message to the configured Telegram chat."""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message
            }
            response = requests.post(url, json=data)
            print(f"TELEGRAM DEBUG RESPONSE: {response.status_code} {response.text}")
            if response.status_code == 200:
                logger.info(f"Telegram message sent: {message}")
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def send_startup_message(self):
        """Send initial startup message with trading configuration."""
        start_msg = (
            f"🚀 Trading Bot Started!\n"
            f"Product ID: {TRADING_CONFIG['product_id']}\n"
            f"Initial Capital: ${TRADING_CONFIG['initial_capital']}\n"
            f"Leverage: {TRADING_CONFIG['leverage']}x\n"
            f"Order Block Strategy Params:\n"
            f"- Candle TF: {TRADING_CONFIG['timeframe']}\n"
            f"- Price Move Threshold: {TRADING_CONFIG['sensitivity']*100:.2f}%\n"
            f"- Min Volume Percentile: {TRADING_CONFIG['min_volume_percentile']}\n"
            f"- SMA Trend: 20/50\n"
            f"- Min Candles Between Trades: {TRADING_CONFIG['min_trades_distance']}\n"
            f"- Stop Loss: {TRADING_CONFIG['stop_loss_pct']*100:.2f}%\n"
            f"- Trailing Stop: {TRADING_CONFIG['trailing_stop_pct']*100:.2f}%"
        )
        self.send_message(start_msg)

    def send_trade_alert(self, trade_type, symbol, price, size):
        """
        Send formatted trade alert
        """
        emoji = "🟢" if trade_type.lower() == "buy" else "🔴"
        message = (
            f"{emoji} <b>{trade_type.upper()} Alert</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Price: {price}\n"
            f"Size: {size}"
        )
        return self.send_message(message)

    def send_error_alert(self, error_message):
        """
        Send error alert
        """
        message = f"⚠️ <b>Error Alert</b>\n\n{error_message}"
        return self.send_message(message)

def send_telegram_message(message):
    bot = TelegramBot()
    return bot.send_message(message) 