import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Union, List, Optional
import numpy as np


def create_insider_trading_chart(
    price_series: Union[pd.Series, List[float]], 
    insider_df: pd.DataFrame,
    price_dates: Optional[Union[pd.Series, List]] = None,
    title: str = "Insider Trading Activity vs Price",
    height: int = 600,
    show_volume: bool = False,
    volume_data: Optional[Union[pd.Series, List[float]]] = None
) -> go.Figure:
    """
    Create an interactive Plotly chart showing insider trading operations as dots on a price chart.
    
    Parameters:
    -----------
    price_series : pd.Series or List[float]
        Price data to plot
    insider_df : pd.DataFrame
        DataFrame containing insider trading data with columns:
        - operation_date_parsed: date of the operation
        - operation: type of operation (Acquisition/Vente)
        - quantity: number of shares
        - price_eur: price per share
        - total_value_eur: total transaction value
        - author: person who made the transaction
    price_dates : pd.Series or List, optional
        Dates corresponding to price_series. If None, will use index of price_series
    title : str
        Chart title
    height : int
        Chart height in pixels
    show_volume : bool
        Whether to show volume subplot
    volume_data : pd.Series or List[float], optional
        Volume data to display if show_volume=True
        
    Returns:
    --------
    go.Figure
        Interactive Plotly figure
    """
    
    # Convert inputs to pandas Series if needed
    if isinstance(price_series, list):
        price_series = pd.Series(price_series)
    
    if price_dates is not None:
        if isinstance(price_dates, list):
            price_dates = pd.Series(price_dates)
        price_series.index = price_dates
    
    # Ensure insider_df has required columns
    required_cols = ['operation_date_parsed', 'operation', 'quantity', 'price_eur', 'total_value_eur', 'author']
    missing_cols = [col for col in required_cols if col not in insider_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in insider_df: {missing_cols}")
    
    # Convert operation_date_parsed to datetime if it's not already
    insider_df['operation_date_parsed'] = pd.to_datetime(insider_df['operation_date_parsed'])
    
    # Create subplots
    if show_volume and volume_data is not None:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(title, 'Volume'),
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    # Add price line
    if show_volume and volume_data is not None:
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>'
            )
        )
    
    # Add insider trading dots
    colors = {'Acquisition': '#2ca02c', 'Vente': '#d62728'}
    
    for operation_type in ['Acquisition', 'Vente']:
        mask = insider_df['operation'] == operation_type
        if mask.any():
            subset = insider_df[mask]
            
            if show_volume and volume_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=subset['operation_date_parsed'],
                        y=subset['price_eur'],
                        mode='markers',
                        name=f'{operation_type}',
                        marker=dict(
                            size=subset['quantity'] / subset['quantity'].max() * 20 + 5,
                            color=colors[operation_type],
                            opacity=0.8,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=(
                            '<b>Date:</b> %{x}<br>'
                            '<b>Operation:</b> ' + operation_type + '<br>'
                            '<b>Price:</b> %{y:.2f}€<br>'
                            '<b>Quantity:</b> %{customdata[0]:,.0f}<br>'
                            '<b>Total Value:</b> %{customdata[1]:,.0f}€<br>'
                            '<b>Author:</b> %{customdata[2]}<extra></extra>'
                        ),
                        customdata=subset[['quantity', 'total_value_eur', 'author']].values,
                        row=1, col=1
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=subset['operation_date_parsed'],
                        y=subset['price_eur'],
                        mode='markers',
                        name=f'{operation_type}',
                        marker=dict(
                            size=subset['quantity'] / subset['quantity'].max() * 20 + 5,
                            color=colors[operation_type],
                            opacity=0.8,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=(
                            '<b>Date:</b> %{x}<br>'
                            '<b>Operation:</b> ' + operation_type + '<br>'
                            '<b>Price:</b> %{y:.2f}€<br>'
                            '<b>Quantity:</b> %{customdata[0]:,.0f}<br>'
                            '<b>Total Value:</b> %{customdata[1]:,.0f}€<br>'
                            '<b>Author:</b> %{customdata[2]}<extra></extra>'
                        ),
                        customdata=subset[['quantity', 'total_value_eur', 'author']].values
                    )
                )
    
    # Add volume if requested
    if show_volume and volume_data is not None:
        if isinstance(volume_data, list):
            volume_data = pd.Series(volume_data, index=price_series.index)
        
        fig.add_trace(
            go.Bar(
                x=volume_data.index,
                y=volume_data.values,
                name='Volume',
                marker_color='rgba(128, 128, 128, 0.5)',
                hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        height=height,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            title='Price (€)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    )
    
    if show_volume and volume_data is not None:
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def create_insider_summary_chart(insider_df: pd.DataFrame) -> go.Figure:
    """
    Create a summary chart showing insider trading statistics.
    
    Parameters:
    -----------
    insider_df : pd.DataFrame
        DataFrame containing insider trading data
        
    Returns:
    --------
    go.Figure
        Interactive Plotly figure with summary statistics
    """
    
    # Convert dates
    insider_df['operation_date_parsed'] = pd.to_datetime(insider_df['operation_date_parsed'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Operations by Type',
            'Total Value by Month',
            'Top Insiders by Volume',
            'Price Distribution'
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # 1. Operations by type (pie chart)
    operation_counts = insider_df['operation'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=operation_counts.index,
            values=operation_counts.values,
            name="Operations by Type"
        ),
        row=1, col=1
    )
    
    # 2. Total value by month (bar chart)
    monthly_value = insider_df.groupby(
        insider_df['operation_date_parsed'].dt.to_period('M')
    )['total_value_eur'].sum().reset_index()
    monthly_value['operation_date_parsed'] = monthly_value['operation_date_parsed'].astype(str)
    
    fig.add_trace(
        go.Bar(
            x=monthly_value['operation_date_parsed'],
            y=monthly_value['total_value_eur'],
            name="Monthly Value",
            marker_color='#1f77b4'
        ),
        row=1, col=2
    )
    
    # 3. Top insiders by volume (bar chart)
    top_insiders = insider_df.groupby('author')['quantity'].sum().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(
            x=top_insiders.values,
            y=top_insiders.index,
            orientation='h',
            name="Top Insiders",
            marker_color='#ff7f0e'
        ),
        row=2, col=1
    )
    
    # 4. Price distribution (histogram)
    fig.add_trace(
        go.Histogram(
            x=insider_df['price_eur'],
            nbinsx=20,
            name="Price Distribution",
            marker_color='#2ca02c'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Insider Trading Summary",
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_yaxes(title_text="Total Value (€)", row=1, col=2)
    fig.update_xaxes(title_text="Quantity", row=2, col=1)
    fig.update_xaxes(title_text="Price (€)", row=2, col=2)
    
    return fig


# Example usage function
def example_usage():
    """
    Example of how to use the visualization functions
    """
    import pandas as pd
    
    # Load your data
    insider_df = pd.read_csv('data/insider_trades_1_100_20250727.csv')
    
    # Create sample price data (replace with your actual price data)
    dates = pd.date_range('2025-01-01', '2025-07-27', freq='D')
    np.random.seed(42)
    price_series = pd.Series(
        np.cumsum(np.random.randn(len(dates)) * 0.02) + 100,
        index=dates
    )
    
    # Create the main chart
    fig1 = create_insider_trading_chart(
        price_series=price_series,
        insider_df=insider_df,
        title="Insider Trading Activity - Sample Stock"
    )
    
    # Create summary chart
    fig2 = create_insider_summary_chart(insider_df)
    
    # Display charts
    fig1.show()
    fig2.show()
    
    return fig1, fig2


if __name__ == "__main__":
    example_usage()
