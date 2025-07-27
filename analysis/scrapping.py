import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from typing import Optional
from datetime import datetime

def scrape_insider_trades(start_page: int = 1, end_page: int = 1) -> pd.DataFrame:
    """
    Scrape insider trading data from abcbourse.com
    
    Args:
        start_page (int): Starting page number (default: 1)
        end_page (int): Ending page number (default: 1)
    
    Returns:
        pd.DataFrame: DataFrame containing insider trading data
    """
    
    base_url = "https://www.abcbourse.com/marches/transactions_dirigeants"
    all_trades = []
    
    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    for page in range(start_page, end_page + 1):
        print(f"Scraping page {page}...")
        
        # Construct URL for the current page
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}?page={page}"
        
        try:
            # Make request with retry logic
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        print(f"Failed to fetch page {page} after 3 attempts: {e}")
                        continue
                    time.sleep(2)
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main table
            main_table = soup.find('table', {'id': 'tabQuotes'})
            if not main_table:
                print(f"Could not find main table on page {page}")
                continue
            
            # Get all rows from tbody
            tbody = main_table.find('tbody')
            if not tbody:
                print(f"Could not find tbody on page {page}")
                continue
                
            rows = tbody.find_all('tr')
            
            i = 0
            while i < len(rows):
                row = rows[i]
                
                # Skip detail rows (they have class 'dtlinsider')
                if 'dtlinsider' in row.get('class', []):
                    i += 1
                    continue
                
                # Process main data row
                cells = row.find_all('td')
                if len(cells) < 6:
                    i += 1
                    continue
                
                # Extract basic information from main row
                company_link = cells[0].find('a')
                company = company_link.get_text(strip=True) if company_link else None
                company_href = company_link.get('href', '') if company_link else ''
                
                declaration_date = cells[1].get_text(strip=True)
                operation = cells[2].get_text(strip=True)
                instrument = cells[3].get_text(strip=True)
                amount_str = cells[4].get_text(strip=True)
                
                # Parse amount (remove € and convert to float)
                amount = None
                if amount_str:
                    amount_clean = re.sub(r'[€\s]', '', amount_str).replace(',', '.')
                    try:
                        amount = float(amount_clean)
                    except ValueError:
                        pass
                
                # Check if next row is a detail row
                detail_info = {}
                if i + 1 < len(rows) and 'dtlinsider' in rows[i + 1].get('class', []):
                    detail_row = rows[i + 1]
                    detail_table = detail_row.find('table')
                    
                    if detail_table:
                        detail_rows = detail_table.find_all('tr')
                        
                        # First row contains author information
                        if len(detail_rows) > 0:
                            author_cell = detail_rows[0].find('td')
                            if author_cell:
                                author_text = author_cell.get_text(strip=True)
                                # Remove "Auteur: " prefix
                                author = author_text.replace('Auteur: ', '') if author_text.startswith('Auteur: ') else author_text
                                detail_info['author'] = author
                        
                        # Second row contains operation details
                        if len(detail_rows) > 1:
                            detail_cells = detail_rows[1].find_all('td')
                            
                            for cell in detail_cells:
                                cell_text = cell.get_text(strip=True)
                                
                                # Extract operation date
                                if cell_text.startswith("Date d'opération:"):
                                    date_str = cell_text.replace("Date d'opération:", "").strip()
                                    detail_info['operation_date'] = date_str
                                
                                # Extract quantity
                                elif cell_text.startswith("Quantité:"):
                                    quantity_str = cell_text.replace("Quantité:", "").strip()
                                    # Remove spaces and convert to int
                                    quantity_clean = re.sub(r'\s', '', quantity_str)
                                    try:
                                        detail_info['quantity'] = int(quantity_clean)
                                    except ValueError:
                                        detail_info['quantity'] = None
                                
                                # Extract price
                                elif cell_text.startswith("Prix:"):
                                    price_str = cell_text.replace("Prix:", "").replace("€", "").strip()
                                    price_clean = price_str.replace(',', '.')
                                    try:
                                        detail_info['price_eur'] = float(price_clean)
                                    except ValueError:
                                        detail_info['price_eur'] = None
                        
                        # Third row might contain comments
                        if len(detail_rows) > 2:
                            comment_cell = detail_rows[2].find('td')
                            if comment_cell:
                                comment_text = comment_cell.get_text(strip=True)
                                if comment_text.startswith("Commentaires:"):
                                    comments = comment_text.replace("Commentaires:", "").strip()
                                    detail_info['comments'] = comments if comments else None
                    
                    # Skip the detail row in next iteration
                    i += 2
                else:
                    i += 1
                
                # Calculate total value
                total_value = None
                if detail_info.get('quantity') and detail_info.get('price_eur'):
                    total_value = detail_info['quantity'] * detail_info['price_eur']
                
                # Create trade record
                trade_record = {
                    'company': company,
                    'company_href': company_href,
                    'declaration_date': declaration_date,
                    'operation': operation,
                    'instrument': instrument,
                    'amount_from_main': amount,
                    'author': detail_info.get('author'),
                    'operation_date': detail_info.get('operation_date'),
                    'quantity': detail_info.get('quantity'),
                    'price_eur': detail_info.get('price_eur'),
                    'total_value_eur': total_value,
                    'comments': detail_info.get('comments'),
                    'scraped_page': page,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Only add if we have essential data
                if company and detail_info.get('author'):
                    all_trades.append(trade_record)
            
            # Add delay between requests to be respectful
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing page {page}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_trades)
    
    # Convert operation_date to datetime if possible
    if not df.empty and 'operation_date' in df.columns:
        try:
            df['operation_date_parsed'] = pd.to_datetime(df['operation_date'], format='%d/%m/%Y', errors='coerce')
        except:
            pass
    
    # Convert declaration_date to datetime if possible
    if not df.empty and 'declaration_date' in df.columns:
        try:
            df['declaration_date_parsed'] = pd.to_datetime(df['declaration_date'], format='%d/%m/%Y', errors='coerce')
        except:
            pass
    
    print(f"Successfully scraped {len(df)} trades from {end_page - start_page + 1} pages")
    
    return df

def save_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """
    Save DataFrame to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str, optional): Custom filename. If None, generates timestamp-based name
    
    Returns:
        str: Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'insider_trades_{timestamp}.csv'
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved to {filename}")
    return filename

def display_summary(df: pd.DataFrame) -> None:
    """
    Display summary statistics of the scraped data
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
    """
    if df.empty:
        print("No data to summarize")
        return
    
    print(f"\n=== INSIDER TRADING DATA SUMMARY ===")
    print(f"Total trades: {len(df)}")
    
    if 'operation_date_parsed' in df.columns:
        valid_dates = df['operation_date_parsed'].dropna()
        if not valid_dates.empty:
            print(f"Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
    
    print(f"Unique companies: {df['company'].nunique()}")
    print(f"Unique authors: {df['author'].nunique()}")
    
    if 'total_value_eur' in df.columns and df['total_value_eur'].notna().any():
        total_value = df['total_value_eur'].sum()
        avg_value = df['total_value_eur'].mean()
        print(f"Total transaction value: {total_value:,.2f} EUR")
        print(f"Average transaction value: {avg_value:,.2f} EUR")
    
    print(f"\nTop 5 companies by number of trades:")
    print(df['company'].value_counts().head())
    
    print(f"\nTop 5 operation types:")
    print(df['operation'].value_counts().head())

def get_pagination_info(soup: BeautifulSoup) -> int:
    """
    Extract pagination information from the soup
    
    Args:
        soup: BeautifulSoup object of the page
        
    Returns:
        int: Total number of pages available
    """
    pagination = soup.find('ul', {'class': 'pagin'})
    if not pagination:
        return 1
    
    page_links = pagination.find_all('a')
    if not page_links:
        return 1
    
    max_page = 1
    for link in page_links:
        href = link.get('href', '')
        if 'page=' in href:
            try:
                page_num = int(href.split('page=')[1])
                max_page = max(max_page, page_num)
            except:
                continue
    
    return max_page

# Example usage and testing function
def test_scraper():
    """Test the scraper with a small sample"""
    print("Testing scraper with page 1...")
    df = scrape_insider_trades(1, 1)
    
    if not df.empty:
        print("✅ Scraper working correctly!")
        print(f"Scraped {len(df)} trades")
        print("\nSample data:")
        print(df[['company', 'author', 'operation_date', 'operation', 'quantity', 'price_eur']].head())
    else:
        print("❌ Scraper returned empty DataFrame")
    
    return df

# Example usage
if __name__ == "__main__":
    # Test the scraper first
    test_df = test_scraper()
    
    if not test_df.empty:
        # If test successful, scrape more pages
        print("\n" + "="*50)
        print("Scraping multiple pages...")
        trades_df = scrape_insider_trades(start_page=1, end_page=3)
        
        # Display summary
        display_summary(trades_df)
        
        # Save to CSV
        if not trades_df.empty:
            csv_file = save_to_csv(trades_df)
            
            # Display first few rows
            print(f"\nFirst 5 records:")
            print(trades_df.head())
            
            # Show column info
            print(f"\nDataFrame info:")
            print(trades_df.info())