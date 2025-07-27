import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_insider_opportunities(df: pd.DataFrame) -> dict:
    """
    Comprehensive analysis of insider trading data to identify buying opportunities
    
    Args:
        df: DataFrame from the scraper containing insider trades
        
    Returns:
        dict: Dictionary containing various analyses and opportunities
    """
    
    # Ensure we have the required columns
    required_cols = ['company', 'operation', 'author', 'quantity', 'price_eur', 'total_value_eur']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean and prepare data
    df_clean = df.dropna(subset=['company', 'operation', 'quantity', 'price_eur']).copy()
    
    # Standardize operation types
    df_clean['operation_clean'] = df_clean['operation'].str.lower()
    df_clean['is_buy'] = df_clean['operation_clean'].str.contains('acquisition|achat|souscription', na=False)
    df_clean['is_sell'] = df_clean['operation_clean'].str.contains('cession|vente', na=False)
    
    # Parse dates
    if 'operation_date_parsed' not in df_clean.columns and 'operation_date' in df_clean.columns:
        df_clean['operation_date_parsed'] = pd.to_datetime(df_clean['operation_date'], format='%d/%m/%Y', errors='coerce')
    
    analyses = {}
    
    # 1. CLUSTER ANALYSIS - Multiple insiders buying same stock
    analyses['cluster_buying'] = find_cluster_buying(df_clean)
    
    # 2. VOLUME ANALYSIS - Large purchases by insiders
    analyses['large_purchases'] = find_large_purchases(df_clean)
    
    # 3. EXECUTIVE LEVEL ANALYSIS - Focus on high-level executives
    analyses['executive_buying'] = find_executive_buying(df_clean)
    
    # 4. RECENT ACTIVITY ANALYSIS - Focus on recent trades
    analyses['recent_activity'] = find_recent_activity(df_clean)
    
    # 5. BUY/SELL RATIO ANALYSIS - Companies with strong buy signals
    analyses['buy_sell_ratio'] = calculate_buy_sell_ratios(df_clean)
    
    # 6. REPEATED BUYER ANALYSIS - Insiders who keep buying
    analyses['repeated_buyers'] = find_repeated_buyers(df_clean)
    
    # 7. SECTOR ANALYSIS - Hot sectors
    analyses['sector_trends'] = analyze_sector_trends(df_clean)
    
    # 8. FINAL RECOMMENDATIONS
    analyses['recommendations'] = generate_recommendations(analyses)
    
    return analyses

def find_cluster_buying(df: pd.DataFrame) -> pd.DataFrame:
    """Find companies where multiple insiders are buying"""
    
    # Filter for buys only
    buys = df[df['is_buy'] == True].copy()
    
    # Group by company and count unique buyers
    cluster_stats = buys.groupby('company').agg({
        'author': 'nunique',
        'total_value_eur': ['sum', 'mean', 'count'],
        'operation_date_parsed': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    cluster_stats.columns = ['company', 'unique_buyers', 'total_value', 'avg_value', 'num_trades', 'first_trade', 'last_trade']
    
    # Filter for companies with multiple buyers
    clusters = cluster_stats[cluster_stats['unique_buyers'] >= 2].sort_values('total_value', ascending=False)
    
    # Add recent activity flag (last 30 days)
    recent_date = datetime.now() - timedelta(days=30)
    clusters['recent_activity'] = clusters['last_trade'] >= recent_date
    
    return clusters

def find_large_purchases(df: pd.DataFrame, percentile=90) -> pd.DataFrame:
    """Find unusually large purchases by value"""
    
    buys = df[df['is_buy'] == True].copy()
    
    # Calculate threshold for large purchases
    threshold = buys['total_value_eur'].quantile(percentile/100)
    
    large_purchases = buys[buys['total_value_eur'] >= threshold].copy()
    large_purchases = large_purchases.sort_values('total_value_eur', ascending=False)
    
    # Add relative size indicator
    large_purchases['value_percentile'] = large_purchases['total_value_eur'].rank(pct=True) * 100
    
    return large_purchases[['company', 'author', 'total_value_eur', 'quantity', 'price_eur', 
                           'operation_date_parsed', 'value_percentile']]

def find_executive_buying(df: pd.DataFrame) -> pd.DataFrame:
    """Focus on purchases by high-level executives"""
    
    # Define executive keywords (case-insensitive)
    executive_keywords = [
        'pdg', 'ceo', 'president', 'directeur general', 'directeur général',
        'administrateur', 'conseil', 'gerant', 'gérant'
    ]
    
    buys = df[df['is_buy'] == True].copy()
    
    # Create executive flag
    buys['is_executive'] = False
    for keyword in executive_keywords:
        buys.loc[buys['author'].str.lower().str.contains(keyword, na=False), 'is_executive'] = True
    
    executive_buys = buys[buys['is_executive'] == True].copy()
    
    # Group by company
    exec_summary = executive_buys.groupby('company').agg({
        'total_value_eur': ['sum', 'count'],
        'author': lambda x: list(set(x)),
        'operation_date_parsed': 'max'
    }).reset_index()
    
    exec_summary.columns = ['company', 'total_exec_value', 'num_exec_trades', 'executives', 'last_trade']
    exec_summary = exec_summary.sort_values('total_exec_value', ascending=False)
    
    return exec_summary

def find_recent_activity(df: pd.DataFrame, days=30) -> pd.DataFrame:
    """Find recent insider buying activity"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    recent_buys = df[
        (df['is_buy'] == True) & 
        (df['operation_date_parsed'] >= cutoff_date)
    ].copy()
    
    if recent_buys.empty:
        return pd.DataFrame()
    
    recent_summary = recent_buys.groupby('company').agg({
        'total_value_eur': 'sum',
        'author': 'nunique',
        'quantity': 'sum',
        'operation_date_parsed': 'max'
    }).reset_index()
    
    recent_summary.columns = ['company', 'recent_buy_value', 'recent_buyers', 'total_shares', 'latest_trade']
    recent_summary = recent_summary.sort_values('recent_buy_value', ascending=False)
    
    return recent_summary

def calculate_buy_sell_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate buy/sell ratios for each company"""
    
    # Separate buys and sells
    buy_summary = df[df['is_buy'] == True].groupby('company')['total_value_eur'].sum()
    sell_summary = df[df['is_sell'] == True].groupby('company')['total_value_eur'].sum()
    
    # Combine and calculate ratios
    ratio_df = pd.DataFrame({
        'buy_value': buy_summary,
        'sell_value': sell_summary
    }).fillna(0)
    
    ratio_df['total_value'] = ratio_df['buy_value'] + ratio_df['sell_value']
    ratio_df['buy_ratio'] = ratio_df['buy_value'] / ratio_df['total_value']
    ratio_df['net_buying'] = ratio_df['buy_value'] - ratio_df['sell_value']
    
    # Filter for companies with significant activity and strong buy ratio
    significant_activity = ratio_df[ratio_df['total_value'] >= ratio_df['total_value'].quantile(0.5)]
    strong_buyers = significant_activity[significant_activity['buy_ratio'] >= 0.7]
    
    return strong_buyers.sort_values('net_buying', ascending=False).reset_index()

def find_repeated_buyers(df: pd.DataFrame) -> pd.DataFrame:
    """Find insiders who repeatedly buy the same stock"""
    
    buys = df[df['is_buy'] == True].copy()
    
    # Group by company and author
    repeat_analysis = buys.groupby(['company', 'author']).agg({
        'total_value_eur': ['sum', 'count'],
        'operation_date_parsed': ['min', 'max']
    }).reset_index()
    
    repeat_analysis.columns = ['company', 'author', 'total_invested', 'num_purchases', 'first_purchase', 'last_purchase']
    
    # Filter for repeated buyers (2+ purchases)
    repeated = repeat_analysis[repeat_analysis['num_purchases'] >= 2]
    
    # Calculate consistency score
    repeated['days_span'] = (repeated['last_purchase'] - repeated['first_purchase']).dt.days
    repeated['consistency_score'] = repeated['num_purchases'] / (repeated['days_span'] + 1) * 100
    
    return repeated.sort_values(['total_invested', 'consistency_score'], ascending=False)

def analyze_sector_trends(df: pd.DataFrame) -> dict:
    """Analyze trends by sector (basic analysis based on company names)"""
    
    # This is a simplified sector analysis
    # In practice, you'd want to map companies to actual sectors
    
    buys = df[df['is_buy'] == True].copy()
    
    sector_activity = buys.groupby('company').agg({
        'total_value_eur': 'sum',
        'author': 'nunique'
    }).reset_index()
    
    sector_activity = sector_activity.sort_values('total_value_eur', ascending=False)
    
    return {
        'top_companies_by_insider_buying': sector_activity.head(10),
        'summary': f"Top {len(sector_activity)} companies with insider buying activity"
    }

def generate_recommendations(analyses: dict) -> pd.DataFrame:
    """Generate final investment recommendations based on all analyses"""
    
    recommendations = []
    
    # Score companies based on multiple factors
    companies_scores = defaultdict(lambda: {'score': 0, 'reasons': []})
    
    # Factor 1: Cluster buying (multiple insiders)
    if not analyses['cluster_buying'].empty:
        for _, row in analyses['cluster_buying'].head(10).iterrows():
            company = row['company']
            companies_scores[company]['score'] += row['unique_buyers'] * 2
            companies_scores[company]['reasons'].append(f"{row['unique_buyers']} different insiders buying")
    
    # Factor 2: Executive buying
    if not analyses['executive_buying'].empty:
        for _, row in analyses['executive_buying'].head(10).iterrows():
            company = row['company']
            companies_scores[company]['score'] += 3
            companies_scores[company]['reasons'].append("Executive-level buying")
    
    # Factor 3: Recent activity
    if not analyses['recent_activity'].empty:
        for _, row in analyses['recent_activity'].head(10).iterrows():
            company = row['company']
            companies_scores[company]['score'] += 2
            companies_scores[company]['reasons'].append("Recent buying activity")
    
    # Factor 4: Strong buy/sell ratio
    if not analyses['buy_sell_ratio'].empty:
        for _, row in analyses['buy_sell_ratio'].head(10).iterrows():
            company = row['company']
            companies_scores[company]['score'] += row['buy_ratio'] * 2
            companies_scores[company]['reasons'].append(f"Strong buy ratio ({row['buy_ratio']:.1%})")
    
    # Convert to DataFrame
    for company, data in companies_scores.items():
        recommendations.append({
            'company': company,
            'opportunity_score': data['score'],
            'reasons': '; '.join(data['reasons'])
        })
    
    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        rec_df = rec_df.sort_values('opportunity_score', ascending=False)
    
    return rec_df

def create_analysis_report(analyses: dict) -> str:
    """Create a formatted text report of the analysis"""
    
    report = []
    report.append("="*60)
    report.append("INSIDER TRADING ANALYSIS REPORT")
    report.append("="*60)
    report.append("")
    
    # Top Recommendations
    if not analyses['recommendations'].empty:
        report.append("🎯 TOP INVESTMENT OPPORTUNITIES:")
        report.append("-" * 40)
        for i, row in analyses['recommendations'].head(5).iterrows():
            report.append(f"{i+1}. {row['company']}")
            report.append(f"   Score: {row['opportunity_score']:.1f}")
            report.append(f"   Reasons: {row['reasons']}")
            report.append("")
    
    # Cluster Buying
    if not analyses['cluster_buying'].empty:
        report.append("👥 CLUSTER BUYING (Multiple Insiders):")
        report.append("-" * 40)
        for i, row in analyses['cluster_buying'].head(5).iterrows():
            report.append(f"• {row['company']}: {row['unique_buyers']} buyers, €{row['total_value']:,.0f} total")
    
    report.append("")
    
    # Recent Activity
    if not analyses['recent_activity'].empty:
        report.append("🔥 RECENT ACTIVITY (Last 30 days):")
        report.append("-" * 40)
        for i, row in analyses['recent_activity'].head(5).iterrows():
            report.append(f"• {row['company']}: €{row['recent_buy_value']:,.0f} by {row['recent_buyers']} insiders")
    
    return "\n".join(report)

# Example usage function
def run_complete_analysis(df: pd.DataFrame):
    """Run the complete analysis and display results"""
    
    print("🔍 Running insider trading analysis...")
    
    try:
        analyses = analyze_insider_opportunities(df)
        
        # Display summary report
        report = create_analysis_report(analyses)
        print(report)
        
        # Return analyses for further use
        return analyses
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

# Data visualization functions
def plot_analysis_charts(analyses: dict):
    """Create visualization charts for the analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Insider Trading Analysis Dashboard', fontsize=16)
    
    # Chart 1: Top companies by cluster buying
    if not analyses['cluster_buying'].empty:
        top_clusters = analyses['cluster_buying'].head(10)
        axes[0,0].barh(top_clusters['company'], top_clusters['total_value'])
        axes[0,0].set_title('Companies with Multiple Insider Buyers')
        axes[0,0].set_xlabel('Total Value (EUR)')
    
    # Chart 2: Buy/Sell ratios
    if not analyses['buy_sell_ratio'].empty:
        ratios = analyses['buy_sell_ratio'].head(10)
        axes[0,1].scatter(ratios['buy_value'], ratios['seSll_value'])
        axes[0,1].set_title('Buy vs Sell Value by Company')
        axes[0,1].set_xlabel('Buy Value (EUR)')
        axes[0,1].set_ylabel('Sell Value (EUR)')
    
    # Chart 3: Recent activity
    if not analyses['recent_activity'].empty:
        recent = analyses['recent_activity'].head(10)
        axes[1,0].bar(range(len(recent)), recent['recent_buy_value'])
        axes[1,0].set_title('Recent Buying Activity (Last 30 days)')
        axes[1,0].set_ylabel('Value (EUR)')
        axes[1,0].set_xticks(range(len(recent)))
        axes[1,0].set_xticklabels(recent['company'], rotation=45, ha='right')
    
    # Chart 4: Opportunity scores
    if not analyses['recommendations'].empty:
        recs = analyses['recommendations'].head(10)
        axes[1,1].barh(recs['company'], recs['opportunity_score'])
        axes[1,1].set_title('Investment Opportunity Scores')
        axes[1,1].set_xlabel('Opportunity Score')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Insider Trading Analysis Module")
    print("Usage: analyses = run_complete_analysis(your_scraped_dataframe)")