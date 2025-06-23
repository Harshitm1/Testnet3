import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy import OrderBlocks
from delta_rest_client import DeltaRestClient
from config import TRADING_CONFIG, API_CONFIG
from logger import setup_logger
import traceback
from telegram_bot import TelegramBot, send_telegram_message
import requests
import matplotlib.pyplot as plt

logger = setup_logger('trading_bot')

class TradingBot:
    def __init__(self):
        """Initialize the trading bot with configuration."""
        self.client = DeltaRestClient(
            base_url=API_CONFIG['base_url'],
            api_key=API_CONFIG['api_key'],
            api_secret=API_CONFIG['api_secret']
        )
        
        # Initialize trading parameters from config
        self.symbol = TRADING_CONFIG['symbol']
        self.product_id = TRADING_CONFIG['product_id']
        self.timeframe = TRADING_CONFIG['timeframe']
        self.capital = TRADING_CONFIG['initial_capital']
        self.leverage = TRADING_CONFIG['leverage']
        
        # Risk management parameters
        self.max_loss_per_trade = 0.02  # 2% max loss per trade
        self.trailing_stop_pct = 0.015  # 1.5% trailing stop
        
        # Initialize strategy
        self.strategy = OrderBlocks(
            sensitivity=TRADING_CONFIG['sensitivity'],
            min_volume_percentile=TRADING_CONFIG['min_volume_percentile'],
            trend_period=TRADING_CONFIG['trend_period'],
            min_trades_distance=TRADING_CONFIG['min_trades_distance']
        )
        
        # Initialize Telegram bot
        self.telegram = TelegramBot()
        
        logger.info(f"Successfully initialized trading for product ID: {self.product_id}")
        self.telegram.send_startup_message()
        
        # Trading state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.stop_loss = None
        self.trailing_stop = None
        self.position_size = 0
        
        # Performance tracking
        self.trades = []
        self.equity = []
        
    def rate_limit_aware_call(self, func, *args, **kwargs):
        """Call an API function, handle 429 errors and sleep as needed."""
        while True:
            try:
                return func(*args, **kwargs)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    reset_ms = int(e.response.headers.get('X-RATE-LIMIT-RESET', '1000'))
                    reset_sec = max(reset_ms / 1000, 1)
                    logger.warning(f"Rate limit hit. Sleeping for {reset_sec:.1f} seconds.")
                    time.sleep(reset_sec)
                else:
                    raise

    def fetch_latest_data(self):
        """Fetch historical data from Delta Exchange."""
        try:
            # Get more historical data (500 candles) for proper indicator calculation
            candles = self.client.get_history_candles(
                symbol=self.symbol,
                resolution=self.timeframe,
                limit=500
            )
            
            if not candles:
                logger.error("No candles received from API")
                return None
            
            # Convert to DataFrame and process data
            df = pd.DataFrame(candles)
            
            # Standardize column names if needed
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp and set as index
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} historical candles")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
            
    def calculate_position_size(self, capital, entry_price):
        """Calculate position size based on current capital and entry price."""
        try:
            # Calculate position size using entire current capital
            position_size = (capital * self.leverage) / entry_price
            
            # Allow fractional units for more accurate trading
            position_size = round(position_size, 8)  # Round to 8 decimal places
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0  # Return 0 on error to prevent trading

    def place_order(self, order):
        """Place an order with proper parameters."""
        try:
            # Ensure positive order size and handle fractional units
            size = abs(float(order['size']))
            if size < 0.00000001:  # Minimum size check
                logger.error("Order size too small")
                return None
            
            # Convert order parameters to match Delta Exchange API
            api_order = {
                'product_id': self.product_id,
                'size': size,
                'side': order['side'],
                'order_type': 'market_order'
            }
            
            # Add reduce_only flag for closing orders
            if order.get('reduce_only', False):
                api_order['reduce_only'] = 'true'
            
            # Place the order
            result = self.rate_limit_aware_call(self.client.create_order, api_order)
            
            # Send Telegram notification
            self.telegram.send_trade_alert(order['side'], self.symbol, order.get('price', 'MARKET'), size)
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
            
    def update_position(self, current_price):
        """Update position and check for exits based on trailing stop"""
        if self.position != 0:
            try:
                # Calculate unrealized PnL
                if self.position == 1:  # Long position
                    current_value = self.position_size * current_price
                    unrealized_pnl = current_value - (self.position_size * self.entry_price)
                    
                    # Update trailing stop
                    new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
                    if new_trailing_stop > self.trailing_stop:
                        self.trailing_stop = new_trailing_stop
                        logger.info(f"Updated long trailing stop to ${self.trailing_stop:.2f}")
                    
                    # Check for stop loss
                    if current_price <= self.trailing_stop:
                        self.close_position('trailing_stop', current_price, unrealized_pnl)
                        
                else:  # Short position
                    current_value = self.position_size * self.entry_price
                    unrealized_pnl = current_value - (self.position_size * current_price)
                    
                    # Update trailing stop
                    new_trailing_stop = current_price * (1 + self.trailing_stop_pct)
                    if new_trailing_stop < self.trailing_stop:
                        self.trailing_stop = new_trailing_stop
                        logger.info(f"Updated short trailing stop to ${self.trailing_stop:.2f}")
                
                    # Check for stop loss
                    if current_price >= self.trailing_stop:
                        self.close_position('trailing_stop', current_price, unrealized_pnl)
                        
            except Exception as e:
                logger.error(f"Error updating position: {str(e)}")

    def close_position(self, reason, exit_price, pnl):
        """Close the current position and update capital"""
        try:
            order = {
                'product_id': self.product_id,
                'size': self.position_size,
                'side': 'sell' if self.position == 1 else 'buy',
                'order_type': 'market_order',
                'reduce_only': 'true'
            }
            
            if self.client.create_order(order):
                # Update capital with realized PnL
                self.capital += pnl
                
                # Record trade
                self.trades.append({
                    'type': 'long' if self.position == 1 else 'short',
                    'entry_price': self.entry_price,
                    'entry_time': self.entry_time,
                    'exit_price': exit_price,
                    'exit_time': datetime.now(),
                    'pnl': pnl,
                    'exit_reason': reason,
                    'return_pct': (pnl / self.entry_capital) * 100
                })
                
                logger.info(f"Closed position at ${exit_price:.2f}, PnL: ${pnl:.2f}, Reason: {reason}")
                self.telegram.send_trade_exit_alert(exit_price, pnl, reason)
                
                # Reset position
                self.position = 0
                self.entry_price = None
                self.stop_loss = None
                self.trailing_stop = None
                self.position_size = 0
                
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            
    def execute_trades(self, df):
        """Execute trades based on order block signals"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Check for new trade entry or opposite signal exit
            for bull_box in self.strategy.bull_boxes:
                if bull_box['start_idx'] == len(df) - 1:  # New signal
                    if self.position == 0:  # New long entry
                        self.position = 1
                        self.entry_price = current_price
                        self.entry_time = datetime.now()
                        self.entry_capital = self.capital
                        
                        # Calculate position size using current capital
                        self.position_size = self.calculate_position_size(self.capital, current_price)
                        
                        # Set initial stop loss
                        self.stop_loss = current_price * (1 - self.max_loss_per_trade)
                        self.trailing_stop = self.stop_loss
                        
                        # Place entry order
                        order = {
                            'product_id': self.product_id,
                            'size': self.position_size,
                            'side': 'buy',
                            'order_type': 'market_order'
                        }
                        
                        if self.client.create_order(order):
                            logger.info(f"Opened LONG at {current_price:.2f} with size {self.position_size}")
                            self.telegram.send_trade_alert('BUY', self.symbol, 'MARKET', self.position_size)
                            
                    elif self.position == -1:  # Exit short on opposite signal
                        current_value = self.position_size * self.entry_price
                        pnl = current_value - (self.position_size * current_price)
                        self.close_position('opposite_signal', current_price, pnl)

            for bear_box in self.strategy.bear_boxes:
                if bear_box['start_idx'] == len(df) - 1:  # New signal
                    if self.position == 0:  # New short entry
                        self.position = -1
                        self.entry_price = current_price
                        self.entry_time = datetime.now()
                        self.entry_capital = self.capital
                        
                        # Calculate position size using current capital
                        self.position_size = self.calculate_position_size(self.capital, current_price)
                        
                        # Set initial stop loss
                        self.stop_loss = current_price * (1 + self.max_loss_per_trade)
                        self.trailing_stop = self.stop_loss
                        
                        # Place entry order
                        order = {
                            'product_id': self.product_id,
                            'size': self.position_size,
                            'side': 'sell',
                            'order_type': 'market_order'
                        }
                        
                        if self.client.create_order(order):
                            logger.info(f"Opened SHORT at {current_price:.2f} with size {self.position_size}")
                            self.telegram.send_trade_alert('SELL', self.symbol, 'MARKET', self.position_size)
                            
                    elif self.position == 1:  # Exit long on opposite signal
                        current_value = self.position_size * current_price
                        pnl = current_value - (self.position_size * self.entry_price)
                        self.close_position('opposite_signal', current_price, pnl)
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            
    def print_wallet_balance(self):
        try:
            for asset_id in ['USDT', 'ETH']:
                balance = self.rate_limit_aware_call(self.client.get_balances, asset_id)
                print(f'WALLET DEBUG RESPONSE for {asset_id}:', balance)
                if balance:
                    print(f"Wallet Balance ({asset_id}): {balance['balance']}")
                else:
                    print(f"Could not fetch wallet balance for {asset_id}")
        except Exception as e:
            print(f"Error fetching wallet balance: {e}")
            
    def run(self):
        """Main trading loop"""
        logger.info("Starting trading bot...")
        
        while True:
            try:
                # Fetch latest data
                df = self.fetch_latest_data()
                if df is None or df.empty:
                    logger.error("No data fetched. Retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                # Calculate indicators and find order blocks
                df = self.strategy.find_order_blocks(df)
                
                # Update current position if any
                current_price = df['close'].iloc[-1]
                self.update_position(current_price)
                
                # Execute new trades
                self.execute_trades(df)
                
                # Update equity curve
                if self.position == 0:
                    self.equity.append(self.capital)
                else:
                    # Calculate unrealized PnL
                    if self.position == 1:
                        current_value = self.position_size * current_price
                        unrealized_pnl = current_value - (self.position_size * self.entry_price)
                    else:
                        current_value = self.position_size * self.entry_price
                        unrealized_pnl = current_value - (self.position_size * current_price)
                    self.equity.append(self.capital + unrealized_pnl)
                
                # Sleep until next candle
                time.sleep(60)  # 1-minute timeframe
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}\n{traceback.format_exc()}")
                time.sleep(60)  # Wait before retrying

    def test_proxy_order(self):
        print("\n--- Proxy Buy Order Test ---")
        order = {
            'product_id': self.product_id,
            'size': 1,
            'side': 'buy',
            'order_type': 'market_order'
        }
        result = self.client.create_order(order)
        print("Buy order result:", result)
        self.telegram.send_trade_alert('buy', self.symbol, 'TEST', 1)
        print("\n--- Proxy Sell Order Test ---")
        order = {
            'product_id': self.product_id,
            'size': 1,
            'side': 'sell',
            'order_type': 'market_order'
        }
        result = self.client.create_order(order)
        print("Sell order result:", result)
        self.telegram.send_trade_alert('sell', self.symbol, 'TEST', 1)

    def plot_live_results(self, df, bull_boxes, bear_boxes, equity_curve):
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        ax1.plot(df['timestamp'], df['close'], label='Price', color='blue', alpha=0.5)
        for b in bear_boxes:
            ax1.axvspan(df['timestamp'].iloc[b['start_idx']], df['timestamp'].iloc[-1],
                        ymin=(b['bot'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        ymax=(b['top'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        color='#506CD3', alpha=0.33)
        for b in bull_boxes:
            ax1.axvspan(df['timestamp'].iloc[b['start_idx']], df['timestamp'].iloc[-1],
                        ymin=(b['bot'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        ymax=(b['top'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        color='#64C4AC', alpha=0.33)
        ax1.set_title('ETH-USD Price and Order Blocks (Live)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        # Equity curve
        if equity_curve:
            times = [e['timestamp'] for e in equity_curve]
            values = [e['equity'] for e in equity_curve]
            ax2.plot(times, values, label='Equity', color='purple', linewidth=2)
        ax2.set_title('Equity Curve (Live)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Equity ($)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig('live_trading_ob_equity.png')
        plt.close()

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
    # bot.test_proxy_order() 
