"""
Portfolio analytics module for performance analysis across multiple time horizons.
Designed for portfolio managers to evaluate model performance in current market conditions.
Enhanced with Plotly interactive visualizations.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import logging
from scipy import stats
import os

# Configure logging
logger = logging.getLogger(__name__)

# Set default plotly template
pio.templates.default = "plotly_white"

class PortfolioAnalytics:
    """Class for portfolio manager-focused performance analytics and visualizations."""
    
    def __init__(self, results_df, benchmark_returns=None, figsize=(900, 600), theme="plotly_white"):
        """
        Initialize the portfolio analytics.
        
        Args:
            results_df (pd.DataFrame): DataFrame with backtest results
            benchmark_returns (pd.Series, optional): Series with benchmark returns
            figsize (tuple): Figure size (width, height) in pixels
            theme (str): Plotly theme/template
        """
        self.results_df = results_df
        self.benchmark_returns = benchmark_returns
        self.figsize = figsize
        self.theme = theme
        
        # Set the theme
        pio.templates.default = self.theme
        
        # Align benchmark with results if provided
        if self.benchmark_returns is not None:
            if isinstance(self.benchmark_returns, pd.Series):
                self.benchmark_returns = self.benchmark_returns.reindex(self.results_df.index)
                # Calculate benchmark cumulative returns
                self.benchmark_cumulative = (1 + self.benchmark_returns.fillna(0)).cumprod()
            else:
                # Convert to pandas Series if it's a numpy array
                self.benchmark_returns = pd.Series(
                    self.benchmark_returns, 
                    index=self.results_df.index[:len(self.benchmark_returns)]
                )
                self.benchmark_cumulative = (1 + self.benchmark_returns.fillna(0)).cumprod()
        
        # Prepare data
        self._prepare_data()
        
        logger.info(f"Initialized PortfolioAnalytics with data from {self.results_df.index[0]} to {self.results_df.index[-1]}")
    
    def _prepare_data(self):
        """Prepare data for performance analysis."""
        # Ensure we have portfolio values
        if 'portfolio_value' not in self.results_df.columns:
            if 'cumulative_return' in self.results_df.columns:
                self.results_df['portfolio_value'] = (1 + self.results_df['cumulative_return'])
        
        # Ensure we have returns
        if 'return' not in self.results_df.columns:
            self.results_df['return'] = self.results_df['portfolio_value'].pct_change().fillna(0)
        
        # Calculate drawdown if not present
        if 'drawdown' not in self.results_df.columns:
            running_max = self.results_df['portfolio_value'].cummax()
            self.results_df['drawdown'] = (self.results_df['portfolio_value'] / running_max - 1)
        
        # Create time period columns for analysis
        # We'll identify the current date as the last date in the dataset
        current_date = self.results_df.index[-1]
        
        # Calculate time period filters
        self.week_mask = (self.results_df.index >= current_date - pd.Timedelta(days=7))
        self.two_weeks_mask = (self.results_df.index >= current_date - pd.Timedelta(days=14))
        self.month_mask = (self.results_df.index >= current_date - pd.Timedelta(days=30))
        self.six_months_mask = (self.results_df.index >= current_date - pd.Timedelta(days=180))
        self.year_mask = (self.results_df.index >= current_date - pd.Timedelta(days=365))
        self.two_years_mask = (self.results_df.index >= current_date - pd.Timedelta(days=730))
        
        # Create period names
        self.periods = {
            '1W': self.week_mask,
            '2W': self.two_weeks_mask,
            '1M': self.month_mask,
            '6M': self.six_months_mask,
            '1Y': self.year_mask,
            '2Y': self.two_years_mask,
            'All': pd.Series(True, index=self.results_df.index)
        }
        
        # Calculate metrics for each period
        self.period_metrics = self._calculate_period_metrics()
    
    def _calculate_period_metrics(self):
        """Calculate performance metrics for each time period."""
        period_metrics = {}
        
        for period_name, mask in self.periods.items():
            # Skip periods with insufficient data
            if sum(mask) <= 1:
                continue
                
            period_data = self.results_df[mask]
            returns = period_data['return']
            
            # Performance metrics
            total_return = period_data['portfolio_value'].iloc[-1] / period_data['portfolio_value'].iloc[0] - 1
            
            # Calculate annualization factor based on period length
            days = (period_data.index[-1] - period_data.index[0]).days
            if days < 1:  # Avoid division by zero
                days = 1
            annualization_factor = 252 / days * len(period_data)
            
            # Risk metrics
            if len(returns) > 1:
                volatility = returns.std() * np.sqrt(annualization_factor)
                sharpe = (returns.mean() * annualization_factor) / (returns.std() * np.sqrt(annualization_factor)) if returns.std() > 0 else 0
                max_drawdown = period_data['drawdown'].min()
                win_rate = (returns > 0).mean()
                
                # Calculate benchmark metrics if available
                if self.benchmark_returns is not None:
                    benchmark_returns = self.benchmark_returns[mask]
                    benchmark_total_return = (1 + benchmark_returns.fillna(0)).prod() - 1
                    
                    if len(benchmark_returns.dropna()) > 1:
                        benchmark_volatility = benchmark_returns.std() * np.sqrt(annualization_factor)
                        benchmark_sharpe = (benchmark_returns.mean() * annualization_factor) / (benchmark_returns.std() * np.sqrt(annualization_factor)) if benchmark_returns.std() > 0 else 0
                        excess_return = total_return - benchmark_total_return
                        
                        # Calculate beta and alpha
                        covariance = np.cov(returns.fillna(0), benchmark_returns.fillna(0))[0, 1]
                        variance = np.var(benchmark_returns.fillna(0))
                        beta = covariance / variance if variance > 0 else 0
                        alpha = total_return - beta * benchmark_total_return
                    else:
                        benchmark_volatility = 0
                        benchmark_sharpe = 0
                        excess_return = 0
                        beta = 0
                        alpha = 0
                else:
                    benchmark_total_return = 0
                    benchmark_volatility = 0
                    benchmark_sharpe = 0
                    excess_return = 0
                    beta = 0
                    alpha = 0
            else:
                volatility = 0
                sharpe = 0
                max_drawdown = 0
                win_rate = 0
                benchmark_total_return = 0
                benchmark_volatility = 0
                benchmark_sharpe = 0
                excess_return = 0
                beta = 0
                alpha = 0
            
            period_metrics[period_name] = {
                'total_return': total_return,
                'annualized_return': total_return * annualization_factor,
                'volatility': volatility,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'benchmark_return': benchmark_total_return,
                'excess_return': excess_return,
                'beta': beta,
                'alpha': alpha,
                'n_days': len(period_data)
            }
        
        return period_metrics
    
    def create_period_performance_dashboard(self, save_path=None, show_plot=True, show=True):
        """
        Create a performance dashboard showing key metrics across time periods.
        
        Args:
            save_path (str, optional): Path to save the figure
            show_plot (bool): Whether to display the plot (deprecated, use show)
            show (bool): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance by Time Horizon',
                'Risk-Adjusted Performance',
                'Relative Performance vs Benchmark',
                'Maximum Drawdown by Time Horizon'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.10
        )

        # Add main title
        fig.update_layout(
            title={
                'text': 'Performance Analysis Across Time Horizons',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            height=self.figsize[1],
            width=self.figsize[0],
            showlegend=True
        )
        
        # 1. Multi-period performance chart
        self._plot_period_performance(fig, row=1, col=1)
        
        # 2. Risk-adjusted metrics
        self._plot_risk_metrics(fig, row=1, col=2)
        
        # 3. Relative performance vs benchmark
        self._plot_relative_performance(fig, row=2, col=1)
        
        # 4. Drawdown comparison
        self._plot_drawdown_comparison(fig, row=2, col=2)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            fig.write_html(save_path.replace('.png', '.html'))
            logger.info(f"Period performance dashboard saved to {save_path}")
        
        # Show plot if requested (support both legacy show_plot and new show parameter)
        if show_plot or show:
            fig.show()
        
        return fig
    
    def _plot_period_performance(self, fig, row=1, col=1):
        """Plot performance metrics across different time periods."""
        periods = []
        returns = []
        vols = []
        
        for period, metrics in self.period_metrics.items():
            periods.append(period)
            returns.append(metrics['total_return'] * 100)
            vols.append(metrics['volatility'] * 100)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Period': periods,
            'Return (%)': returns,
            'Volatility (%)': vols
        })
        
        # Sort by period length (custom order)
        period_order = ['1W', '2W', '1M', '6M', '1Y', '2Y', 'All']
        df['Period_Sorted'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period_Sorted')
        
        # Plot returns as bars
        fig.add_trace(
            go.Bar(
                x=df['Period'],
                y=df['Return (%)'],
                name='Return (%)',
                marker_color='#0066CC',
                text=df['Return (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Create second y-axis for volatility
        fig.add_trace(
            go.Bar(
                x=df['Period'],
                y=df['Volatility (%)'],
                name='Volatility (%)',
                marker_color='#CC0000',
                opacity=0.7,
                yaxis='y2',
                text=df['Volatility (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Add horizontal line at y=0 for returns
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=0,
            xref="paper", yref="y1",
            line=dict(color="black", width=1),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Return (%)',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Volatility (%)',
            overlaying='y',
            side='right',
            row=row, col=col
        )
    
    def _plot_risk_metrics(self, fig, row=1, col=1):
        """Plot risk-adjusted metrics across different time periods."""
        periods = []
        sharpes = []
        win_rates = []
        
        for period, metrics in self.period_metrics.items():
            periods.append(period)
            sharpes.append(metrics['sharpe'])
            win_rates.append(metrics['win_rate'] * 100)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Period': periods,
            'Sharpe Ratio': sharpes,
            'Win Rate (%)': win_rates
        })
        
        # Sort by period length (custom order)
        period_order = ['1W', '2W', '1M', '6M', '1Y', '2Y', 'All']
        df['Period_Sorted'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period_Sorted')
        
        # Plot Sharpe Ratio as bars
        fig.add_trace(
            go.Bar(
                x=df['Period'],
                y=df['Sharpe Ratio'],
                name='Sharpe Ratio',
                marker_color='#0066CC',
                text=df['Sharpe Ratio'].apply(lambda x: f'{x:.2f}'),
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Create second y-axis for win rate
        fig.add_trace(
            go.Bar(
                x=df['Period'],
                y=df['Win Rate (%)'],
                name='Win Rate (%)',
                marker_color='green',
                opacity=0.7,
                yaxis='y2',
                text=df['Win Rate (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Add horizontal line at y=0 for Sharpe
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=0,
            xref="paper", yref="y1",
            line=dict(color="black", width=1),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Sharpe Ratio',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Win Rate (%)',
            overlaying='y',
            side='right',
            row=row, col=col
        )
    
    def _plot_relative_performance(self, fig, row=1, col=1):
        """Plot relative performance vs benchmark across different time periods."""
        # Skip if no benchmark is available
        if self.benchmark_returns is None:
            fig.add_annotation(
                x=0.5, y=0.5,
                text='No benchmark data available',
                showarrow=False,
                font=dict(size=15),
                xref="paper", yref="paper",
                row=row, col=col
            )
            return
            
        periods = []
        alpha_values = []
        excess_returns = []
        
        for period, metrics in self.period_metrics.items():
            periods.append(period)
            alpha_values.append(metrics['alpha'] * 100)
            excess_returns.append(metrics['excess_return'] * 100)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Period': periods,
            'Alpha (%)': alpha_values,
            'Excess Return (%)': excess_returns
        })
        
        # Sort by period length (custom order)
        period_order = ['1W', '2W', '1M', '6M', '1Y', '2Y', 'All']
        df['Period_Sorted'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period_Sorted')
        
        # Plot excess return as bars
        fig.add_trace(
            go.Bar(
                x=df['Period'],
                y=df['Excess Return (%)'],
                name='Excess Return (%)',
                marker_color='#0066CC',
                text=df['Excess Return (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Create second y-axis for alpha
        fig.add_trace(
            go.Bar(
                x=df['Period'],
                y=df['Alpha (%)'],
                name='Alpha (%)',
                marker_color='purple',
                opacity=0.7,
                yaxis='y2',
                text=df['Alpha (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Add horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=0,
            xref="paper", yref="y1",
            line=dict(color="black", width=1),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Excess Return (%)',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Alpha (%)',
            overlaying='y',
            side='right',
            row=row, col=col
        )
    
    def _plot_drawdown_comparison(self, fig, row=1, col=1):
        """Plot maximum drawdown across different time periods."""
        periods = []
        max_drawdowns = []
        
        for period, metrics in self.period_metrics.items():
            periods.append(period)
            max_drawdowns.append(metrics['max_drawdown'] * 100)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Period': periods,
            'Max Drawdown (%)': max_drawdowns
        })
        
        # Sort by period length (custom order)
        period_order = ['1W', '2W', '1M', '6M', '1Y', '2Y', 'All']
        df['Period_Sorted'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period_Sorted')
        
        # Plot max drawdown as bars
        fig.add_trace(
            go.Bar(
                x=df['Period'],
                y=df['Max Drawdown (%)'],
                name='Max Drawdown (%)',
                marker_color='#CC0000',
                text=df['Max Drawdown (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Max Drawdown (%)',
            autorange="reversed",  # Invert y-axis to show drawdowns as negative
            row=row, col=col
        )
    
    def create_pnl_time_horizon_dashboard(self, save_path=None, show_plot=True, show=True):
        """
        Create a PnL dashboard showing performance over different time horizons.
        
        Args:
            save_path (str, optional): Path to save the figure
            show_plot (bool): Whether to display the plot (deprecated, use show)
            show (bool): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Define the periods to display
        display_periods = [
            ('1 Week', self.week_mask),
            ('2 Weeks', self.two_weeks_mask),
            ('1 Month', self.month_mask),
            ('6 Months', self.six_months_mask),
            ('1 Year', self.year_mask),
            ('2 Years', self.two_years_mask)
        ]
        
        # Create figure with subplots - increase vertical spacing
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[title for title, _ in display_periods],
            vertical_spacing=0.2,  # Increased from 0.15
            horizontal_spacing=0.12  # Increased from 0.10
        )

        # Add main title
        fig.update_layout(
            title={
                'text': 'PnL Analysis Across Time Horizons',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            height=900,  # Increased from 800 for more spacing
            width=self.figsize[0],
            showlegend=True
        )
        
        # Create a subplot for each time period
        for i, (title, mask) in enumerate(display_periods):
            row, col = divmod(i, 2)
            row += 1  # 1-based indexing
            col += 1  # 1-based indexing
            self._plot_period_pnl(fig, mask, title, row, col)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            fig.write_html(save_path.replace('.png', '.html'))
            logger.info(f"PnL time horizon dashboard saved to {save_path}")
        
        # Show plot if requested (support both legacy show_plot and new show parameter)
        if show_plot or show:
            fig.show()
        
        return fig
    
    def _plot_period_pnl(self, fig, mask, title, row, col):
        """Plot PnL for a specific time period."""
        # Get data for the specified period
        period_data = self.results_df[mask].copy()
        
        # Skip if insufficient data
        if len(period_data) <= 1:
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f'Insufficient data for {title}',
                showarrow=False,
                font=dict(size=15),
                xref=f"x{row}{col}", yref=f"y{row}{col}",
                row=row, col=col
            )
            return
        
        # Normalize to start at 100%
        start_value = period_data['portfolio_value'].iloc[0]
        period_data['normalized_value'] = period_data['portfolio_value'] / start_value * 100
        
        # Plot portfolio value
        fig.add_trace(
            go.Scatter(
                x=period_data.index,
                y=period_data['normalized_value'],
                mode='lines',
                name='Strategy',
                line=dict(color='#0066CC', width=2),
                showlegend=(row == 1 and col == 1)  # Only show legend for first subplot
            ),
            row=row, col=col
        )
        
        # Add benchmark if available
        if self.benchmark_returns is not None:
            benchmark_period = self.benchmark_returns[mask]
            if len(benchmark_period) > 0:
                benchmark_cumulative = (1 + benchmark_period.fillna(0)).cumprod()
                benchmark_normalized = benchmark_cumulative / benchmark_cumulative.iloc[0] * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_normalized.index,
                        y=benchmark_normalized,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='#999999', width=1.5, dash='dash'),
                        showlegend=(row == 1 and col == 1)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        
        # Calculate total return for the period
        total_return = period_data['normalized_value'].iloc[-1] / period_data['normalized_value'].iloc[0] - 1
        
        # Create text annotations as a single annotation block in the corner
        annotation_text = []
        
        # Add Strategy return
        annotation_text.append(f"Strategy: {total_return:.2%}")
        
        # Add Benchmark and Alpha if available
        if self.benchmark_returns is not None and len(benchmark_period) > 0:
            benchmark_return = benchmark_normalized.iloc[-1] / benchmark_normalized.iloc[0] - 1
            annotation_text.append(f"Benchmark: {benchmark_return:.2%}")
            
            # Add Alpha
            alpha = total_return - benchmark_return
            alpha_color = 'green' if alpha > 0 else 'red'
            annotation_text.append(f"Alpha: {alpha:.2%}")
        
        # Add the consolidated annotation
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            text="<br>".join(annotation_text),
            align="left",
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            xanchor="left",
            yanchor="bottom",
            row=row,
            col=col
        )
        
        # Format axes - explicitly set the x-axis range
        fig.update_xaxes(
            title_text='Date',
            range=[period_data.index[0], period_data.index[-1]],  # Force x-axis range to match the data
            row=row, 
            col=col
        )
        
        fig.update_yaxes(
            title_text='Value (%)',
            row=row, 
            col=col
        )
    
    def create_return_distribution_dashboard(self, save_path=None, show_plot=True, show=True):
        """
        Create a dashboard showing return distributions for different time periods.
        
        Args:
            save_path (str, optional): Path to save the figure
            show_plot (bool): Whether to display the plot (deprecated, use show)
            show (bool): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Define periods to analyze
        periods = [
            ('Daily', 1, None),
            ('Weekly', 5, 'W'),
            ('Monthly', 21, 'M'),
            ('Quarterly', 63, 'Q')
        ]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{title} Return Distribution' for title, _, _ in periods],
            vertical_spacing=0.15,
            horizontal_spacing=0.10
        )

        # Add main title
        fig.update_layout(
            title={
                'text': 'Return Distribution Analysis',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            height=self.figsize[1],
            width=self.figsize[0],
            showlegend=True
        )
        
        # Create a subplot for each period
        for i, (title, window, freq) in enumerate(periods):
            row, col = divmod(i, 2)
            row += 1  # 1-based indexing
            col += 1  # 1-based indexing
            
            if i == 0:  # Daily returns
                self._plot_return_distribution(fig, title, row, col)
            else:
                self._plot_rolling_return_distribution(fig, title, window, freq, row, col)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            fig.write_html(save_path.replace('.png', '.html'))
            logger.info(f"Return distribution dashboard saved to {save_path}")
        
        # Show plot if requested (support both legacy show_plot and new show parameter)
        if show_plot or show:
            fig.show()
        
        return fig
    
    def _plot_return_distribution(self, fig, period_type='Daily', row=1, col=1):
        """Plot return distribution for daily returns."""
        returns = self.results_df['return'] * 100  # Convert to percentage
        
        # Calculate statistics
        mean = returns.mean()
        std = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        min_ret = returns.min()
        max_ret = returns.max()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Returns',
                nbinsx=30,
                opacity=0.7,
                marker_color='#0066CC',
                histnorm='probability density'
            ),
            row=row, col=col
        )
        
        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, mean, std)
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Dist.',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        # Add vertical line at mean
        fig.add_shape(
            type="line",
            x0=mean, y0=0,
            x1=mean, y1=max(stats.norm.pdf(mean, mean, std) * 1.1, 0.1),
            line=dict(color="red", width=2),
            row=row, col=col
        )
        
        # Add vertical line at zero
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=0, y1=max(stats.norm.pdf(0, mean, std) * 1.1, 0.1),
            line=dict(color="black", width=1, dash="dash"),
            row=row, col=col
        )
        
        # Add statistics annotation
        stats_text = (
            f"Mean: {mean:.2f}%<br>"
            f"Std Dev: {std:.2f}%<br>"
            f"Skew: {skew:.2f}<br>"
            f"Kurtosis: {kurt:.2f}<br>"
            f"Min: {min_ret:.2f}%<br>"
            f"Max: {max_ret:.2f}%"
        )
        
        fig.add_annotation(
            x=0.05, y=0.95,
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor="white",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            xref="paper", yref="paper",
            xanchor='left', yanchor='top',
            row=row, col=col
        )
        
        # Format axes
        fig.update_xaxes(
            title_text=f'{period_type} Return (%)',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Density',
            row=row, col=col
        )
    
    def _plot_rolling_return_distribution(self, fig, period_type, window, freq, row=1, col=1):
        """Plot return distribution for rolling period returns."""
        # Calculate rolling period returns
        if freq == 'W':
            # Weekly returns (approximated as rolling 5-day returns)
            period_returns = self.results_df['portfolio_value'].pct_change(window) * 100
        elif freq == 'M':
            # Monthly returns (approximated as rolling 21-day returns)
            period_returns = self.results_df['portfolio_value'].pct_change(window) * 100
        elif freq == 'Q':
            # Quarterly returns (approximated as rolling 63-day returns)
            period_returns = self.results_df['portfolio_value'].pct_change(window) * 100
        
        # Drop NaN values
        period_returns = period_returns.dropna()
        
        # Calculate statistics
        mean = period_returns.mean()
        std = period_returns.std()
        skew = stats.skew(period_returns)
        kurt = stats.kurtosis(period_returns)
        min_ret = period_returns.min()
        max_ret = period_returns.max()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=period_returns,
                name=f'{period_type} Returns',
                nbinsx=30,
                opacity=0.7,
                marker_color='#0066CC',
                histnorm='probability density'
            ),
            row=row, col=col
        )
        
        # Add normal distribution for comparison
        x = np.linspace(
            max(period_returns.min(), mean - 3*std), 
            min(period_returns.max(), mean + 3*std), 
            100
        )
        y = stats.norm.pdf(x, mean, std)
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Dist.',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False  # Don't show legend for every subplot
            ),
            row=row, col=col
        )
        
        # Add vertical line at mean
        fig.add_shape(
            type="line",
            x0=mean, y0=0,
            x1=mean, y1=max(stats.norm.pdf(mean, mean, std) * 1.1, 0.1),
            line=dict(color="red", width=2),
            row=row, col=col
        )
        
        # Add vertical line at zero
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=0, y1=max(stats.norm.pdf(0, mean, std) * 1.1, 0.1),
            line=dict(color="black", width=1, dash="dash"),
            row=row, col=col
        )
        
        # Add statistics annotation
        stats_text = (
            f"Mean: {mean:.2f}%<br>"
            f"Std Dev: {std:.2f}%<br>"
            f"Skew: {skew:.2f}<br>"
            f"Kurtosis: {kurt:.2f}<br>"
            f"Min: {min_ret:.2f}%<br>"
            f"Max: {max_ret:.2f}%"
        )
        
        fig.add_annotation(
            x=0.05, y=0.95,
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor="white",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            xref="paper", yref="paper",
            xanchor='left', yanchor='top',
            row=row, col=col
        )
        
        # Format axes
        fig.update_xaxes(
            title_text=f'{period_type} Return (%)',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Density',
            row=row, col=col
        )
    
    def create_performance_metrics_table(self):
        """
        Create a formatted table of performance metrics across time periods.
        
        Returns:
            pd.DataFrame: Formatted table of performance metrics
        """
        # Prepare data for table
        table_data = []
        
        # Define the periods to include in order
        period_order = ['1W', '2W', '1M', '6M', '1Y', '2Y', 'All']
        
        for period in period_order:
            if period in self.period_metrics:
                metrics = self.period_metrics[period]
                
                row = {
                    'Period': period,
                    'Days': metrics['n_days'],
                    'Return (%)': metrics['total_return'] * 100,
                    'Ann. Return (%)': metrics['annualized_return'] * 100,
                    'Volatility (%)': metrics['volatility'] * 100,
                    'Sharpe': metrics['sharpe'],
                    'Max DD (%)': metrics['max_drawdown'] * 100,
                    'Win Rate (%)': metrics['win_rate'] * 100
                }
                
                # Add benchmark metrics if available
                if self.benchmark_returns is not None:
                    row.update({
                        'Benchmark (%)': metrics['benchmark_return'] * 100,
                        'Excess Return (%)': metrics['excess_return'] * 100,
                        'Alpha (%)': metrics['alpha'] * 100,
                        'Beta': metrics['beta']
                    })
                
                table_data.append(row)
        
        # Create DataFrame
        metrics_table = pd.DataFrame(table_data)
        
        # Format the table
        formatted_table = metrics_table.copy()
        
        # Format numeric columns
        numeric_cols = formatted_table.columns.drop(['Period', 'Days'])
        for col in numeric_cols:
            if 'Sharpe' in col or 'Beta' in col:
                formatted_table[col] = formatted_table[col].map('{:.2f}'.format)
            else:
                formatted_table[col] = formatted_table[col].map('{:.2f}%'.format)
        
        return formatted_table
    
    def save_performance_report(self, output_path):
        """
        Save a comprehensive performance report as CSV and HTML.
        
        Args:
            output_path (str): Path to save the report
            
        Returns:
            tuple: Paths to the saved CSV and HTML files
        """
        metrics_table = self.create_performance_metrics_table()
        
        # Save as CSV
        csv_path = f"{output_path}_metrics.csv"
        metrics_table.to_csv(csv_path, index=False)
        
        # Create an interactive HTML table with Plotly
        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=list(metrics_table.columns),
                    fill_color='#0066CC',
                    font=dict(color='white', size=12),
                    align='center'
                ),
                cells=dict(
                    values=[metrics_table[col] for col in metrics_table.columns],
                    fill_color=[
                        ['#f9f9f9', '#ffffff'] * (len(metrics_table) // 2 + 1)
                    ],
                    align=['left' if col == 'Period' else 'right' for col in metrics_table.columns],
                    font_size=11,
                    height=25
                )
            )
        ])
        
        # Add title and adjust layout
        fig.update_layout(
            title=f"Portfolio Performance Metrics (Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            title_font=dict(size=16),
            width=max(800, 120 * len(metrics_table.columns)),
            height=100 + 35 * len(metrics_table),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Save HTML
        html_path = f"{output_path}_metrics.html"
        fig.write_html(html_path)
        
        logger.info(f"Performance metrics saved to {csv_path} and {html_path}")
        
        return csv_path, html_path