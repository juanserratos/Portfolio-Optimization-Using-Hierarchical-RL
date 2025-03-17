"""
Portfolio analytics module for performance analysis across multiple time horizons.
Designed for portfolio managers to evaluate model performance in current market conditions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter
from datetime import datetime, timedelta
import logging
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioAnalytics:
    """Class for portfolio manager-focused performance analytics and visualizations."""
    
    def __init__(self, results_df, benchmark_returns=None, figsize=(12, 8), dpi=100, style='seaborn-v0_8-whitegrid'):
        """
        Initialize the portfolio analytics.
        
        Args:
            results_df (pd.DataFrame): DataFrame with backtest results
            benchmark_returns (pd.Series, optional): Series with benchmark returns
            figsize (tuple): Figure size
            dpi (int): DPI for the figures
            style (str): Matplotlib style
        """
        self.results_df = results_df
        self.benchmark_returns = benchmark_returns
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Apply style
        plt.style.use(self.style)
        
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
    
    def create_period_performance_dashboard(self, save_path=None, show_plot=True):
        """
        Create a performance dashboard showing key metrics across time periods.
        
        Args:
            save_path (str, optional): Path to save the figure
            show_plot (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # FIXED: Use constrained_layout instead of tight_layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Title the figure
        fig.suptitle('Performance Analysis Across Time Horizons', fontsize=16, y=0.98)
        
        # 1. Multi-period performance chart
        ax_perf = fig.add_subplot(gs[0, 0])
        self._plot_period_performance(ax_perf)
        
        # 2. Risk-adjusted metrics
        ax_risk = fig.add_subplot(gs[0, 1])
        self._plot_risk_metrics(ax_risk)
        
        # 3. Relative performance vs benchmark
        ax_rel = fig.add_subplot(gs[1, 0])
        self._plot_relative_performance(ax_rel)
        
        # 4. Drawdown comparison
        ax_dd = fig.add_subplot(gs[1, 1])
        self._plot_drawdown_comparison(ax_dd)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Period performance dashboard saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def _plot_period_performance(self, ax):
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
        df['Period'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period')
        
        # Plot returns as bars
        df.plot(x='Period', y='Return (%)', kind='bar', ax=ax, color='#0066CC', width=0.4, position=1, legend=True)
        
        # Create twin axis for volatility
        ax2 = ax.twinx()
        df.plot(x='Period', y='Volatility (%)', kind='bar', ax=ax2, color='#CC0000', width=0.4, position=0, alpha=0.7, legend=True)
        
        # Format axes
        ax.set_title('Performance by Time Horizon', fontsize=12)
        ax.set_ylabel('Return (%)')
        ax2.set_ylabel('Volatility (%)')
        
        # Add horizontal line at y=0 for returns
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.get_legend().remove()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0)
    
    def _plot_risk_metrics(self, ax):
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
        df['Period'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period')
        
        # Plot Sharpe Ratio as bars
        df.plot(x='Period', y='Sharpe Ratio', kind='bar', ax=ax, color='#0066CC', width=0.4, position=1, legend=True)
        
        # Create twin axis for win rate
        ax2 = ax.twinx()
        df.plot(x='Period', y='Win Rate (%)', kind='bar', ax=ax2, color='green', width=0.4, position=0, alpha=0.7, legend=True)
        
        # Format axes
        ax.set_title('Risk-Adjusted Performance by Time Horizon', fontsize=12)
        ax.set_ylabel('Sharpe Ratio')
        ax2.set_ylabel('Win Rate (%)')
        
        # Add horizontal line at y=0 for Sharpe
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.get_legend().remove()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0)
    
    def _plot_relative_performance(self, ax):
        """Plot relative performance vs benchmark across different time periods."""
        # Skip if no benchmark is available
        if self.benchmark_returns is None:
            ax.text(0.5, 0.5, 'No benchmark data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Relative Performance vs Benchmark', fontsize=12)
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
        df['Period'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period')
        
        # Plot excess return as bars
        df.plot(x='Period', y='Excess Return (%)', kind='bar', ax=ax, color='#0066CC', width=0.4, position=1, legend=True)
        
        # Create twin axis for alpha
        ax2 = ax.twinx()
        df.plot(x='Period', y='Alpha (%)', kind='bar', ax=ax2, color='purple', width=0.4, position=0, alpha=0.7, legend=True)
        
        # Format axes
        ax.set_title('Relative Performance vs Benchmark', fontsize=12)
        ax.set_ylabel('Excess Return (%)')
        ax2.set_ylabel('Alpha (%)')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.get_legend().remove()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0)
    
    def _plot_drawdown_comparison(self, ax):
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
        df['Period'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        df = df.sort_values('Period')
        
        # Plot max drawdown as bars
        df.plot(x='Period', y='Max Drawdown (%)', kind='bar', ax=ax, color='#CC0000', legend=False)
        
        # Format axes
        ax.set_title('Maximum Drawdown by Time Horizon', fontsize=12)
        ax.set_ylabel('Max Drawdown (%)')
        
        # Invert y-axis to show drawdowns as negative values
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(df['Max Drawdown (%)']):
            ax.text(i, v - 0.5, f'{v:.1f}%', ha='center', fontweight='bold', color='white')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0)
    
    def create_pnl_time_horizon_dashboard(self, save_path=None, show_plot=True):
        """
        Create a PnL dashboard showing performance over different time horizons.
        
        Args:
            save_path (str, optional): Path to save the figure
            show_plot (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # FIXED: Use constrained_layout instead of tight_layout
        fig = plt.figure(figsize=(15, 12), dpi=self.dpi, constrained_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Title the figure
        fig.suptitle('PnL Analysis Across Time Horizons', fontsize=16, y=0.98)
        
        # Define the periods to display
        display_periods = [
            ('1 Week', self.week_mask),
            ('2 Weeks', self.two_weeks_mask),
            ('1 Month', self.month_mask),
            ('6 Months', self.six_months_mask),
            ('1 Year', self.year_mask),
            ('2 Years', self.two_years_mask)
        ]
        
        # Create a subplot for each time period
        for i, (title, mask) in enumerate(display_periods):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, col])
            self._plot_period_pnl(ax, mask, title)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"PnL time horizon dashboard saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def _plot_period_pnl(self, ax, mask, title):
        """Plot PnL for a specific time period."""
        # Get data for the specified period
        period_data = self.results_df[mask].copy()
        
        # Skip if insufficient data
        if len(period_data) <= 1:
            ax.text(0.5, 0.5, f'Insufficient data for {title}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12)
            return
        
        # Normalize to start at 100%
        start_value = period_data['portfolio_value'].iloc[0]
        period_data['normalized_value'] = period_data['portfolio_value'] / start_value * 100
        
        # Plot portfolio value
        period_data['normalized_value'].plot(ax=ax, color='#0066CC', linewidth=2)
        
        # Add benchmark if available
        if self.benchmark_returns is not None:
            benchmark_period = self.benchmark_returns[mask]
            if len(benchmark_period) > 0:
                benchmark_cumulative = (1 + benchmark_period.fillna(0)).cumprod()
                benchmark_normalized = benchmark_cumulative / benchmark_cumulative.iloc[0] * 100
                benchmark_normalized.plot(ax=ax, color='#999999', linewidth=1.5, linestyle='--', label='Benchmark')
        
        # Calculate total return for the period
        total_return = period_data['normalized_value'].iloc[-1] / period_data['normalized_value'].iloc[0] - 1
        
        # Add simple annotations (no arrows)
        if self.benchmark_returns is not None and len(benchmark_period) > 0:
            benchmark_return = benchmark_normalized.iloc[-1] / benchmark_normalized.iloc[0] - 1
            ax.annotate(f'Strategy: {total_return:.2%}', 
                       xy=(0.02, 0.05), xycoords='axes fraction', 
                       fontweight='bold', color='#0066CC')
            ax.annotate(f'Benchmark: {benchmark_return:.2%}', 
                       xy=(0.02, 0.10), xycoords='axes fraction', 
                       fontweight='bold', color='#999999')
            ax.annotate(f'Alpha: {total_return - benchmark_return:.2%}', 
                       xy=(0.02, 0.15), xycoords='axes fraction', 
                       fontweight='bold', color='green' if total_return > benchmark_return else 'red')
        else:
            ax.annotate(f'Return: {total_return:.2%}', 
                       xy=(0.02, 0.05), xycoords='axes fraction', 
                       fontweight='bold', color='#0066CC')
        
        # Format axes
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Value (%)')
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if len(period_data) > 180:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        elif len(period_data) > 60:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        elif len(period_data) > 20:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left')
    
    def create_return_distribution_dashboard(self, save_path=None, show_plot=True):
        """
        Create a dashboard showing return distributions for different time periods.
        
        Args:
            save_path (str, optional): Path to save the figure
            show_plot (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Define periods to analyze
        periods = [
            ('Weekly', 5, 'W'),
            ('Monthly', 21, 'M'),
            ('Quarterly', 63, 'Q')
        ]
        
        # FIXED: Use constrained_layout instead of tight_layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Title the figure
        fig.suptitle('Return Distribution Analysis', fontsize=16, y=0.98)
        
        # 1. Daily returns distribution
        ax_daily = fig.add_subplot(gs[0, 0])
        self._plot_return_distribution(ax_daily, 'Daily')
        
        # 2. Weekly returns distribution
        ax_weekly = fig.add_subplot(gs[0, 1])
        self._plot_rolling_return_distribution(ax_weekly, *periods[0])
        
        # 3. Monthly returns distribution
        ax_monthly = fig.add_subplot(gs[1, 0])
        self._plot_rolling_return_distribution(ax_monthly, *periods[1])
        
        # 4. Quarterly returns distribution
        ax_quarterly = fig.add_subplot(gs[1, 1])
        self._plot_rolling_return_distribution(ax_quarterly, *periods[2])
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Return distribution dashboard saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def _plot_return_distribution(self, ax, period_type='Daily'):
        """Plot return distribution for daily returns."""
        returns = self.results_df['return'] * 100
        
        # Plot histogram with KDE
        sns.histplot(returns, ax=ax, bins=30, kde=True, color='#0066CC', stat='density', alpha=0.7)
        
        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        mean = returns.mean()
        std = returns.std()
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r--', linewidth=1.5, label='Normal Dist.')
        
        # Add summary statistics
        stats_text = (
            f"Mean: {mean:.2f}%\n"
            f"Std Dev: {std:.2f}%\n"
            f"Skew: {stats.skew(returns):.2f}\n"
            f"Kurtosis: {stats.kurtosis(returns):.2f}\n"
            f"Min: {returns.min():.2f}%\n"
            f"Max: {returns.max():.2f}%"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Format axes
        ax.set_title(f'{period_type} Return Distribution', fontsize=12)
        ax.set_xlabel(f'{period_type} Return (%)')
        ax.set_ylabel('Density')
        
        # Add legend
        ax.legend()
        
        # Add vertical line at mean
        ax.axvline(mean, color='red', linestyle='-', linewidth=1, label='Mean')
        
        # Add vertical line at zero
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    def _plot_rolling_return_distribution(self, ax, period_type, window, freq):
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
        
        # Plot histogram with KDE
        sns.histplot(period_returns, ax=ax, bins=30, kde=True, color='#0066CC', stat='density', alpha=0.7)
        
        # Calculate statistics
        mean = period_returns.mean()
        std = period_returns.std()
        skew = stats.skew(period_returns)
        kurt = stats.kurtosis(period_returns)
        min_ret = period_returns.min()
        max_ret = period_returns.max()
        
        # Add normal distribution for comparison
        x = np.linspace(max(period_returns.min(), mean - 3*std), min(period_returns.max(), mean + 3*std), 100)
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r--', linewidth=1.5, label='Normal Dist.')
        
        # Add summary statistics
        stats_text = (
            f"Mean: {mean:.2f}%\n"
            f"Std Dev: {std:.2f}%\n"
            f"Skew: {skew:.2f}\n"
            f"Kurtosis: {kurt:.2f}\n"
            f"Min: {min_ret:.2f}%\n"
            f"Max: {max_ret:.2f}%"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Format axes
        ax.set_title(f'{period_type} Return Distribution', fontsize=12)
        ax.set_xlabel(f'{period_type} Return (%)')
        ax.set_ylabel('Density')
        
        # Add legend
        ax.legend()
        
        # Add vertical line at mean
        ax.axvline(mean, color='red', linestyle='-', linewidth=1, label='Mean')
        
        # Add vertical line at zero
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
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
        
        # Generate HTML content
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Use triple quotes with explicit newlines to avoid formatting issues
        html_content = """<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { text-align: right; padding: 8px; border: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .period-col { text-align: left; font-weight: bold; }
        .positive { color: green; }
        .negative { color: red; }
        h1 { color: #333; }
        .subtitle { font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <h1>Portfolio Performance Metrics</h1>
    <p class="subtitle">Generated on """ + current_date + """</p>
"""
        
        # Convert DataFrame to HTML
        df_html = metrics_table.to_html(index=False, classes='metrics-table')
        
        # Manually add custom formatting for positive/negative values
        for col in ['Return (%)', 'Ann. Return (%)', 'Excess Return (%)', 'Alpha (%)']:
            if col in metrics_table.columns:
                for i, value in enumerate(metrics_table[col]):
                    if isinstance(value, str):
                        value = float(value.replace('%', ''))
                    
                    value_str = f"{value:.2f}%"
                    if value > 0:
                        formatted_value = f'<span class="positive">{value_str}</span>'
                    elif value < 0:
                        formatted_value = f'<span class="negative">{value_str}</span>'
                    else:
                        formatted_value = value_str
                    
                    original_cell = f'<td>{metrics_table[col].iloc[i]}</td>'
                    df_html = df_html.replace(original_cell, f'<td>{formatted_value}</td>')
        
        # Add custom formatting for Period column
        df_html = df_html.replace('<th>Period</th>', '<th class="period-col">Period</th>')
        for period in metrics_table['Period']:
            df_html = df_html.replace(f'<td>{period}</td>', f'<td class="period-col">{period}</td>')
        
        # Finish HTML
        html_content += df_html
        html_content += """
</body>
</html>
"""
        
        # Save HTML
        html_path = f"{output_path}_metrics.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance metrics saved to {csv_path} and {html_path}")
        
        return csv_path, html_path