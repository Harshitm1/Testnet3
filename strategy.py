import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from config import STRATEGY_CONFIG, TRADING_CONFIG
from logger import setup_logger

logger = setup_logger('strategy')

class OrderBlocks:
    def __init__(self, sensitivity=0.015, ob_mitigation='Close', min_volume_percentile=50, trend_period=20, min_trades_distance=10):
        # Initialize with configuration parameters
        self.sensitivity = sensitivity  # 1.5% price movement threshold
        self.ob_mitigation = ob_mitigation
        self.buy_alert = True
        self.sell_alert = True
        self.bear_boxes = []
        self.bull_boxes = []
        self.min_volume_percentile = min_volume_percentile
        self.trend_period = trend_period
        self.min_trades_distance = min_trades_distance
        self.last_trade_index = -self.min_trades_distance
        
        # Initialize state variables
        self.current_position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.stop_loss_price = None
        self.trailing_stop_price = None
        self.position_size = 0
        self.capital = TRADING_CONFIG['initial_capital']

    def calc_indicators(self, df):
        """Calculate technical indicators for analysis"""
        try:
            # Calculate percentage change over 4 bars
            df = df.copy()
            df.loc[:, 'pc'] = (df['open'] - df['open'].shift(4)) / df['open'].shift(4) * 100
            
            # Calculate volume metrics
            df.loc[:, 'volume_ma'] = df['volume'].rolling(window=20).mean()
            df.loc[:, 'volume_percentile'] = df['volume'].rolling(window=50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan
            )
            
            # Calculate trend indicators
            df.loc[:, 'sma20'] = df['close'].rolling(window=self.trend_period).mean()
            df.loc[:, 'sma50'] = df['close'].rolling(window=50).mean()
            
            # Calculate ATR for volatility filtering
            df.loc[:, 'tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df.loc[:, 'atr'] = df['tr'].rolling(window=14).mean()
            
            # Momentum indicator
            df.loc[:, 'roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
            
            # Add swing high/low detection (fix ambiguity)
            df.loc[:, 'swing_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) &
                                 (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))).astype(int)
            df.loc[:, 'swing_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) &
                                (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))).astype(int)
            
            logger.info("Successfully calculated technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def is_valid_trade_condition(self, df, idx, trade_type='long'):
        """Check if trading conditions are met"""
        try:
            # Check if enough distance from last trade
            if idx - self.last_trade_index < self.min_trades_distance:
                return False

            # Check volume conditions
            volume_percentile = df['volume_percentile'].iloc[idx]
            if pd.isna(volume_percentile) or volume_percentile < self.min_volume_percentile:
                return False

            # Check trend conditions
            sma20 = df['sma20'].iloc[idx]
            sma50 = df['sma50'].iloc[idx]
            roc = df['roc'].iloc[idx]
            
            # If indicators are not yet calculated, allow the trade
            if pd.isna(sma20) or pd.isna(sma50) or pd.isna(roc):
                return True
                
            if trade_type == 'long':
                if not (sma20 > sma50):  # Removed ROC condition to be less restrictive
                    return False
            else:  # short
                if not (sma20 < sma50):  # Removed ROC condition to be less restrictive
                    return False

            # Volatility check - made less restrictive
            current_atr = df['atr'].iloc[idx]
            avg_atr = df['atr'].rolling(window=20).mean().iloc[idx]
            if pd.isna(current_atr) or pd.isna(avg_atr):
                return True
                
            if current_atr > avg_atr * 2:  # Increased threshold from 1.5 to 2
                return False

            return True
            
        except Exception as e:
            logger.error(f"Error checking trade conditions: {str(e)}")
            return False

    def find_order_blocks(self, df):
        try:
            df = self.calc_indicators(df)
            
            # Get the starting index for processing new candles
            start_idx = 4  # Default to start from beginning
            if self.bull_boxes or self.bear_boxes:
                # If we have existing boxes, start from the last processed index
                last_bull_idx = max([b['start_idx'] for b in self.bull_boxes]) if self.bull_boxes else -1
                last_bear_idx = max([b['start_idx'] for b in self.bear_boxes]) if self.bear_boxes else -1
                start_idx = max(4, max(last_bull_idx, last_bear_idx))
            
            for idx in range(start_idx, len(df)):
                pc = df['pc'].iloc[idx]
                prev_pc = df['pc'].iloc[idx-1]
                
                # Skip if pc or prev_pc is not a valid number
                if pd.isna(pc) or pd.isna(prev_pc):
                    continue
                    
                # Check for bearish order block - removed position check
                if (prev_pc > -self.sensitivity and pc <= -self.sensitivity and 
                    not any(idx - b['start_idx'] <= 5 for b in self.bear_boxes)):
                    
                    # Only add if trade conditions are met
                    if self.is_valid_trade_condition(df, idx, 'short'):
                        for i in range(idx-4, max(idx-16, -1), -1):
                            close_price = df['close'].iloc[i]
                            open_price = df['open'].iloc[i]
                            
                            if pd.isna(close_price) or pd.isna(open_price):
                                continue
                                
                            if close_price > open_price:
                                high_price = df['high'].iloc[i]
                                low_price = df['low'].iloc[i]
                                
                                if pd.isna(high_price) or pd.isna(low_price):
                                    continue
                                    
                                self.bear_boxes.append({
                                    'start_idx': i,
                                    'top': high_price,
                                    'bot': low_price
                                })
                                self.last_trade_index = idx
                                break

                # Check for bullish order block - removed position check
                if (prev_pc < self.sensitivity and pc >= self.sensitivity and 
                    not any(idx - b['start_idx'] <= 5 for b in self.bull_boxes)):
                    
                    # Only add if trade conditions are met
                    if self.is_valid_trade_condition(df, idx, 'long'):
                        for i in range(idx-4, max(idx-16, -1), -1):
                            close_price = df['close'].iloc[i]
                            open_price = df['open'].iloc[i]
                            
                            if pd.isna(close_price) or pd.isna(open_price):
                                continue
                                
                            if close_price < open_price:
                                high_price = df['high'].iloc[i]
                                low_price = df['low'].iloc[i]
                                
                                if pd.isna(high_price) or pd.isna(low_price):
                                    continue
                                    
                                self.bull_boxes.append({
                                    'start_idx': i,
                                    'top': high_price,
                                    'bot': low_price
                                })
                                self.last_trade_index = idx
                                break
            
            return df
            
        except Exception as e:
            logger.error(f"Error finding order blocks: {str(e)}")
            return df

    def update_position(self, signal):
        """Update internal position tracking based on signal"""
        try:
            if signal['type'] == 'entry':
                self.current_position = 1 if signal['direction'] == 'long' else -1
                self.entry_price = signal['price']
                self.stop_loss_price = signal['stop_loss']
                self.trailing_stop_price = signal['trailing_stop']
                logger.info(f"Entered {signal['direction']} position at {signal['price']}")
            
            elif signal['type'] == 'exit':
                self.current_position = 0
                self.entry_price = None
                self.stop_loss_price = None
                self.trailing_stop_price = None
                logger.info(f"Exited position at {signal['price']} due to {signal['reason']}")
                
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")

def fetch_market_data(symbol=TRADING_CONFIG['symbol'], timeframe=TRADING_CONFIG['timeframe']):
    """Fetch market data from Delta Exchange"""
    try:
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (60 * 24 * 60 * 60)  # Last 60 days
        
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        resolution = timeframe_map.get(timeframe, '1m')
        
        logger.info(f"Fetching {symbol} data for {timeframe} timeframe")
        
        response = requests.get(
            'https://api.india.delta.exchange/v2/history/candles',
            params={
                'resolution': resolution,
                'symbol': symbol,
                'start': str(start_time),
                'end': str(end_time)
            },
            headers={'Accept': 'application/json'}
        )
        
        response.raise_for_status()
        data = response.json()
        
        if not data or 'result' not in data:
            raise ValueError("No data received from API")
            
        df = pd.DataFrame(data['result'])
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.sort_values('timestamp')
        
        logger.info(f"Successfully fetched {len(df)} candles")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame() 