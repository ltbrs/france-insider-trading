"""
French Stock Ticker Data Analysis

This module provides functions to fetch historical stock data for French stocks,
including daily prices, volume, and other market data.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def get_stock_history(
    ticker: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Get historical stock data for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AIR.PA' for Airbus, 'MC.PA' for LVMH)
        start_date (str or datetime, optional): Start date for data retrieval
        end_date (str or datetime, optional): End date for data retrieval
        period (str): Data period if start/end dates not provided ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        pd.DataFrame: Historical stock data with columns:
            - Open: Opening price
            - High: Highest price of the day
            - Low: Lowest price of the day
            - Close: Closing price
            - Adj Close: Adjusted closing price
            - Volume: Trading volume
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data
        if start_date and end_date:
            # Convert to datetime if strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            hist = stock.history(start=start_date, end=end_date, interval=interval)
        else:
            hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        return hist
    
    except Exception as e:
        logging.error(f"Error retrieving data for {ticker}: {str(e)}")
        return pd.DataFrame()


def get_daily_prices(
    ticker: str,
    days: int = 365
) -> pd.DataFrame:
    """
    Get daily closing prices for a stock over a specified number of days.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to retrieve (default: 365)
        include_volume (bool): Whether to include volume data
    
    Returns:
        pd.DataFrame: Daily price data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    hist = get_stock_history(ticker, start_date=start_date, end_date=end_date)
    return hist


def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Get basic information about a stock.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Stock information including name, sector, market cap, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key information
        key_info = {
            'ticker': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'current_price': info.get('currentPrice', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'currency': info.get('currency', 'N/A'),
            'exchange': info.get('exchange', 'N/A')
        }
        
        return key_info
    
    except Exception as e:
        logging.error(f"Error retrieving info for {ticker}: {str(e)}")
        #raise ValueError(f"Error retrieving info for {ticker}: {str(e)}")


def get_multiple_stocks_history(
    tickers: list,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    period: str = "1y"
) -> Dict[str, pd.DataFrame]:
    """
    Get historical data for multiple stocks simultaneously.
    
    Args:
        tickers (list): List of stock ticker symbols
        start_date (str or datetime, optional): Start date
        end_date (str or datetime, optional): End date
        period (str): Data period if start/end dates not provided
    
    Returns:
        dict: Dictionary with ticker as key and DataFrame as value
    """
    results = {}
    
    for ticker in tickers:
        try:
            results[ticker] = get_stock_history(ticker, start_date, end_date, period)
        except Exception as e:
            print(f"Warning: Could not retrieve data for {ticker}: {str(e)}")
            results[ticker] = pd.DataFrame()
    
    return results


def get_price_change(
    ticker: str,
    days: int = 30
) -> Dict[str, float]:
    """
    Calculate price change statistics for a stock.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to analyze
    
    Returns:
        dict: Price change statistics
    """
    hist = get_daily_prices(ticker, days=days, include_volume=False)
    
    if hist.empty:
        raise ValueError(f"No data available for {ticker}")
    
    current_price = hist['Close'].iloc[-1]
    start_price = hist['Close'].iloc[0]
    
    price_change = current_price - start_price
    price_change_pct = (price_change / start_price) * 100
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'start_price': start_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'days_analyzed': days
    }


# Common French stock tickers
FRENCH_STOCKS = {
    'AIR.PA': 'Airbus SE',
    'MC.PA': 'LVMH Moët Hennessy Louis Vuitton',
    'OR.PA': 'L\'Oréal SA',
    'ASML.AS': 'ASML Holding NV',
    'BNP.PA': 'BNP Paribas SA',
    'CA.PA': 'Carrefour SA',
    'CAP.PA': 'Capgemini SE',
    'DG.PA': 'Vinci SA',
    'ENGI.PA': 'Engie SA',
    'EN.PA': 'Bouygues SA',
    'FP.PA': 'TotalEnergies SE',
    'FTE.PA': 'Orange SA',
    'GLE.PA': 'Société Générale SA',
    'KER.PA': 'Kering SA',
    'ML.PA': 'Michelin SA',
    'ORA.PA': 'Orange SA',
    'PUB.PA': 'Publicis Groupe SA',
    'RNO.PA': 'Renault SA',
    'SAF.PA': 'Safran SA',
    'SAN.PA': 'Sanofi SA',
    'SGO.PA': 'Saint-Gobain SA',
    'SOLB.PA': 'Solvay SA',
    'SU.PA': 'Schneider Electric SE',
    'TEP.PA': 'Teleperformance SE',
    'TTE.PA': 'TotalEnergies SE',
    'UG.PA': 'Peugeot SA',
    'VIV.PA': 'Vivendi SA'
}


def list_french_stocks() -> Dict[str, str]:
    """
    Get a list of common French stock tickers and their names.
    
    Returns:
        dict: Dictionary mapping ticker symbols to company names
    """
    return FRENCH_STOCKS.copy()


# Example usage and testing
if __name__ == "__main__":
    # Example: Get Airbus stock data for the last year
    try:
        print("Fetching Airbus (AIR.PA) stock data for the last year...")
        airbus_data = get_stock_history('AIR.PA', period='1y')
        print(f"Retrieved {len(airbus_data)} days of data")
        print("\nLatest 5 days:")
        print(airbus_data.tail())
        
        # Get stock info
        print("\nStock Information:")
        info = get_stock_info('AIR.PA')
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Get price change
        print("\nPrice Change Analysis (30 days):")
        change = get_price_change('AIR.PA', days=30)
        for key, value in change.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error in example: {e}")
