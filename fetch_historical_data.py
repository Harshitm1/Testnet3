import pandas as pd
import time
from datetime import datetime, timedelta
import requests
from config import API_CONFIG, TRADING_CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fetch_historical_data')

def fetch_historical_data(days=180):
    """Fetch historical data from Delta Exchange."""
    logger.info(f"Fetching {days} days of historical data...")
    
    # Calculate timestamps for past data
    end_time = int(datetime(2024, 1, 1).timestamp())  # Use January 1st, 2024 as end date
    start_time = end_time - (days * 24 * 60 * 60)  # Go back 180 days from there
    
    logger.info(f"Start time: {datetime.fromtimestamp(start_time)}")
    logger.info(f"End time: {datetime.fromtimestamp(end_time)}")
    
    all_candles = []
    current_start = start_time
    batch_size = 1000  # Number of candles per request
    
    try:
        while current_start < end_time:
            logger.info(f"Fetching batch starting at {datetime.fromtimestamp(current_start)}")
            
            # Calculate batch end time
            batch_end = min(current_start + (batch_size * 60), end_time)  # 60 seconds per candle
            
            # Make API request to live API
            url = "https://api.delta.exchange/v2/history/candles"
            params = {
                'symbol': 'ETHUSD',  # Use live symbol
                'resolution': '1m',  # 1-minute candles
                'start': current_start,
                'end': batch_end
            }
            
            logger.info(f"Making request to {url} with params: {params}")
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and data['result']:
                    candles = data['result']
                    all_candles.extend(candles)
                    logger.info(f"Received {len(candles)} candles")
                    
                    # Update start time for next batch
                    current_start = batch_end
                    
                    # Add small delay to avoid rate limits
                    time.sleep(1)
                else:
                    logger.error(f"No data in response: {data}")
                    break
            else:
                logger.error(f"Error fetching data: Status {response.status_code}")
                logger.error(f"Response: {response.text}")
                break
        
        if not all_candles:
            logger.error("No data collected")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp and set as index
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the function
    df = fetch_historical_data(days=180)
    if df is not None:
        print(f"Fetched {len(df)} candles")
        print("\nFirst few candles:")
        print(df.head())
        print("\nLast few candles:")
        print(df.tail()) 