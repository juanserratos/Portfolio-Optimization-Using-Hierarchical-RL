"""
Fixed visualization module for the deep RL trading framework.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import calendar
from datetime import datetime
import logging
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

class BacktestVisualizer:
    """Class for creating enhanced visualizations of backtest results."""
    
    def __init__(self, results_df, benchmark_returns=None, figsize=(12, 8), dpi=100, style='seaborn-v0_8-whitegrid'):
        """
        Initialize the backtest visualizer.
        
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
        
        # Prepare benchmark data if provided
        if self.benchmark_returns is not None:
            if isinstance(self.benchmark_returns, pd.Series):
                self.benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            else:
                # Convert to pandas Series if it's a numpy array or list
                self.benchmark_cumulative = (1 + pd.Series(
                    self.benchmark_returns,
                    index=self.results_df.index[:len(self.benchmark_returns)]
                )).cumprod()
        
        # Calculate additional metrics
        self._calculate_additional_metrics()
        
        logger.info(f"Initialized BacktestVisualizer with data from {self.results_df.index[0]} to {self.results_df.index[-1]}")
    
    def _calculate_additional_metrics(self):
        """Calculate additional metrics for visualization."""
        # Convert portfolio value to cumulative return if not already
        if 'cumulative_return' not in self.results_df.columns:
            if 'portfolio_value' in self.results_df.columns:
                self.results_df['cumulative_return'] = self.results_df['portfolio_value'] / self.results_df['portfolio_value'].iloc[0] - 1
        
        # Calculate returns if not already present
        if 'return' not in self.results_df.columns:
            if 'portfolio_value' in self.results_df.columns:
                self.results_df['return'] = self.results_df['portfolio_value'].pct_change().fillna(0)
        
        # Calculate drawdowns if not present
        if 'drawdown' not in self.results_df.columns:
            if 'cumulative_return' in self.results_df.columns:
                # Calculate running maximum
                running_max = (1 + self.results_df['cumulative_return']).cummax()
                # Calculate drawdown
                self.results_df['drawdown'] = (1 + self.results_df['cumulative_return']) / running_max - 1
        
        # Calculate rolling metrics
        window = min(252, len(self.results_df) // 4)  # Use 252 days or 1/4 of data length
        if window > 20:  # Only calculate if we have enough data
            self.results_df['rolling_volatility'] = self.results_df['return'].rolling(window).std() * np.sqrt(252)
            self.results_df['rolling_return'] = self.results_df['return'].rolling(window).mean() * 252
            self.results_df['rolling_sharpe'] = self.results_df['rolling_return'] / self.results_df['rolling_volatility']
            
        # Add month and year columns for calendar analysis
        self.results_df['year'] = self.results_df.index.year
        self.results_df['month'] = self.results_df.index.month
        
        # Calculate monthly returns
        self.monthly_returns = self.results_df.groupby([self.results_df.index.year, self.results_df.index.month])['return'].apply(
            lambda x: (1 + x).prod() - 1
        ).unstack()
        
    def create_performance_dashboard(self, show_benchmark=True, show_rolling_metrics=True, save_path=None):
        """
        Create a comprehensive performance dashboard.
        
        Args:
            show_benchmark (bool): Whether to show benchmark comparison
            show_rolling_metrics (bool): Whether to show rolling metrics
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Create figure with gridspec for complex layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

        # Title the figure
        fig.suptitle('Trading Strategy Performance Dashboard', fontsize=16, y=0.98)
        
        # 1. Cumulative returns plot
        ax_returns = fig.add_subplot(gs[0, :])
        self._plot_cumulative_returns(ax_returns, show_benchmark=show_benchmark)
        
        # 2. Drawdown plot
        ax_drawdown = fig.add_subplot(gs[1, :])
        self._plot_drawdowns(ax_drawdown)
        
        # 3. Monthly returns heatmap
        ax_monthly = fig.add_subplot(gs[2, 0])
        self._plot_monthly_returns_heatmap(ax_monthly)
        
        # 4. Rolling metrics or stats
        ax_stats = fig.add_subplot(gs[2, 1])
        if show_rolling_metrics and 'rolling_sharpe' in self.results_df.columns:
            self._plot_rolling_metrics(ax_stats)
        else:
            self._plot_return_histogram(ax_stats)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Performance dashboard saved to {save_path}")
        
        return fig
    
    def _plot_cumulative_returns(self, ax, show_benchmark=True):
        """Plot cumulative returns on the given axis."""
        # Plot strategy returns
        self.results_df['cumulative_return'].mul(100).plot(ax=ax, color='#0066CC', linewidth=2, label='Strategy')
        
        # Add benchmark if available
        if show_benchmark and self.benchmark_returns is not None:
            benchmark_cumret = self.benchmark_cumulative - 1
            # Align benchmark to the same time period
            benchmark_cumret = benchmark_cumret.loc[benchmark_cumret.index.intersection(self.results_df.index)]
            benchmark_cumret.mul(100).plot(ax=ax, color='#999999', linewidth=1.5, linestyle='--', label='Benchmark')
        
        # Format the axis
        ax.set_title('Cumulative Returns', fontsize=12)
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Annotate final return
        final_return = self.results_df['cumulative_return'].iloc[-1] * 100
        ax.annotate(f'{final_return:.2f}%', 
                   xy=(self.results_df.index[-1], final_return),
                   xytext=(10, 0), textcoords='offset points',
                   va='center', fontweight='bold')
        
    def _plot_drawdowns(self, ax):
        """Plot drawdowns on the given axis."""
        # Plot drawdowns
        self.results_df['drawdown'].mul(100).plot(ax=ax, color='#CC0000', linewidth=1.5, alpha=0.7)
        
        # Format the axis
        ax.set_title('Drawdowns', fontsize=12)
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage and invert
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.invert_yaxis()
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Highlight maximum drawdown
        max_drawdown_idx = self.results_df['drawdown'].idxmin()
        max_drawdown = self.results_df.loc[max_drawdown_idx, 'drawdown'] * 100
        ax.annotate(f'Max DD: {max_drawdown:.2f}%', 
                   xy=(max_drawdown_idx, max_drawdown),
                   xytext=(10, -20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontweight='bold', color='#CC0000')
    
    def _plot_monthly_returns_heatmap(self, ax):
        """Plot monthly returns heatmap on the given axis."""
        if self.monthly_returns.empty:
            ax.text(0.5, 0.5, 'Insufficient data for monthly heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Returns (%)', fontsize=12)
            return
            
        # Create heatmap
        monthly_returns_pct = self.monthly_returns * 100
        
        # Format month names and handle potentially missing data
        month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
        formatted_data = monthly_returns_pct.rename(columns=month_names)
        
        # Create the heatmap
        sns.heatmap(formatted_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar=False, ax=ax, linewidths=0.5,
                   annot_kws={"size": 9})
        
        # Format the axis
        ax.set_title('Monthly Returns (%)', fontsize=12)
        ax.set_ylabel('Year')
        ax.set_xlabel('')
    
    def _plot_rolling_metrics(self, ax):
        """Plot rolling performance metrics on the given axis."""
        ax2 = ax.twinx()
        
        # Plot rolling Sharpe ratio
        self.results_df['rolling_sharpe'].plot(ax=ax, color='#0066CC', linewidth=1.5, label='Rolling Sharpe')
        
        # Plot rolling volatility on the second y-axis
        (self.results_df['rolling_volatility'] * 100).plot(ax=ax2, color='#CC0000', linewidth=1.5, 
                                                          linestyle='--', label='Rolling Vol')
        
        # Format the axes
        ax.set_title('Rolling Performance Metrics (252-day)', fontsize=12)
        ax.set_ylabel('Sharpe Ratio')
        ax2.set_ylabel('Volatility (%)')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add horizontal line at y=0 for Sharpe
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add grid but only from the first axis
        ax.grid(True, alpha=0.3)
        ax2.grid(False)
    
    def _plot_return_histogram(self, ax):
        """Plot return distribution histogram on the given axis."""
        # Plot histogram
        self.results_df['return'].mul(100).hist(ax=ax, bins=50, alpha=0.7, color='#0066CC')
        
        # Format the axis
        ax.set_title('Return Distribution', fontsize=12)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        
        # Add mean and median lines
        mean_return = self.results_df['return'].mean() * 100
        median_return = self.results_df['return'].median() * 100
        
        ax.axvline(mean_return, color='#CC0000', linestyle='-', linewidth=1.5, label=f'Mean: {mean_return:.2f}%')
        ax.axvline(median_return, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_return:.2f}%')
        
        # Add legend
        ax.legend()
    
    def create_position_analysis(self, num_assets=10, save_path=None):
        """
        Create position and allocation analysis plots.
        
        Args:
            num_assets (int): Number of top assets to display
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Identify weight columns
        weight_cols = [col for col in self.results_df.columns if col.startswith('weight_')]
        
        if not weight_cols:
            logger.warning("No position data found in results DataFrame")
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No position data available', ha='center', va='center')
            return fig
        
        # Create figure with gridspec - FIXED: use constrained_layout instead of tight_layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Title the figure
        fig.suptitle('Portfolio Allocation Analysis', fontsize=16, y=0.98)
        
        # 1. Asset allocation over time
        ax_alloc = fig.add_subplot(gs[0, :])
        self._plot_asset_allocation(ax_alloc, weight_cols)
        
        # 2. Current allocation
        ax_current = fig.add_subplot(gs[1, 0])
        self._plot_current_allocation(ax_current, weight_cols, num_assets)
        
        # 3. Allocation heatmap
        ax_heatmap = fig.add_subplot(gs[1, 1])
        self._plot_allocation_heatmap(ax_heatmap, weight_cols, num_assets)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Position analysis saved to {save_path}")
        
        return fig
    
    def _plot_asset_allocation(self, ax, weight_cols):
        """Plot asset allocation over time."""
        # Create a stacked area chart
        # Use only the top assets to avoid too many colors
        top_assets = self._get_top_assets(weight_cols, n=10)
        allocation_data = self.results_df[top_assets]
        
        # Convert to percentage
        allocation_data = allocation_data.mul(100)
        
        # Create area plot
        allocation_data.plot.area(ax=ax, linewidth=0, alpha=0.7, colormap='tab20')
        
        # Format the axis
        ax.set_title('Asset Allocation Over Time', fontsize=12)
        ax.set_ylabel('Allocation (%)')
        ax.set_xlabel('')
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False,
                 title='Assets', title_fontsize=10)
    
    def _plot_current_allocation(self, ax, weight_cols, num_assets=10):
        """Plot current allocation as a pie chart."""
        # Get the last row of data
        current_weights = self.results_df[weight_cols].iloc[-1]
        
        # Get the top assets
        top_weights = current_weights.abs().nlargest(num_assets)
        top_actual_weights = current_weights[top_weights.index]
        
        # Convert ticker names for display
        labels = [col.replace('weight_', '') for col in top_actual_weights.index]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            top_actual_weights.abs(),
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            textprops={'fontsize': 9},
            colors=plt.cm.tab20.colors[:len(top_actual_weights)]
        )
        
        # Adjust text properties
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
        # Add legend
        ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9)
        
        # Format the axis
        ax.set_title('Current Allocation', fontsize=12)
        ax.set_aspect('equal')
        
    def _plot_allocation_heatmap(self, ax, weight_cols, num_assets=10):
        """Plot allocation heatmap over time."""
        # Get top assets
        top_assets = self._get_top_assets(weight_cols, n=num_assets)
        
        # Resample to monthly data to reduce density
        # FIXED: Change from 'M' to 'ME' to avoid deprecation warning
        monthly_data = self.results_df[top_assets].resample('ME').last()
        
        # Create labels for x-axis (dates)
        date_labels = [d.strftime('%Y-%m') for d in monthly_data.index]
        
        # Create labels for y-axis (tickers)
        ticker_labels = [col.replace('weight_', '') for col in monthly_data.columns]
        
        # Create heatmap with cbar_kws to avoid layout issues
        sns.heatmap(monthly_data.T * 100, cmap='RdBu_r', center=0, 
                   linewidths=0.3, annot=False, fmt='.0f', ax=ax,
                   xticklabels=date_labels, yticklabels=ticker_labels,
                   cbar_kws={'label': 'Allocation (%)', 'use_gridspec': False, 'shrink': 0.9})
        
        # Format the axis
        ax.set_title('Allocation Heatmap (Monthly)', fontsize=12)
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    def _get_top_assets(self, weight_cols, n=10):
        """Get the top n assets by average absolute weight."""
        avg_weights = self.results_df[weight_cols].abs().mean().nlargest(n)
        return avg_weights.index.tolist()
    
    def create_risk_analysis(self, save_path=None):
        """
        Create risk analysis visualizations.
        
        Args:
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # FIXED: Use constrained_layout instead of tight_layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Title the figure
        fig.suptitle('Risk Analysis Dashboard', fontsize=16, y=0.98)
        
        # 1. Rolling volatility
        ax_vol = fig.add_subplot(gs[0, 0])
        self._plot_rolling_volatility(ax_vol)
        
        # 2. Rolling VaR and CVaR
        ax_var = fig.add_subplot(gs[0, 1])
        self._plot_rolling_var(ax_var)
        
        # 3. Drawdown periods
        ax_dd = fig.add_subplot(gs[1, 0])
        self._plot_drawdown_periods(ax_dd)
        
        # 4. Return distribution
        ax_dist = fig.add_subplot(gs[1, 1])
        self._plot_return_distribution(ax_dist)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Risk analysis saved to {save_path}")
        
        return fig
    
    def _plot_rolling_volatility(self, ax):
        """Plot rolling volatility."""
        # Calculate rolling volatility if not already done
        if 'rolling_volatility' not in self.results_df.columns:
            window = min(252, len(self.results_df) // 4)
            self.results_df['rolling_volatility'] = self.results_df['return'].rolling(window).std() * np.sqrt(252)
        
        # Plot rolling volatility
        (self.results_df['rolling_volatility'] * 100).plot(ax=ax, color='#CC0000', linewidth=1.5)
        
        # Format the axis
        ax.set_title('Rolling Volatility (252-day)', fontsize=12)
        ax.set_ylabel('Annualized Volatility (%)')
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add average line
        avg_vol = self.results_df['rolling_volatility'].mean() * 100
        ax.axhline(avg_vol, color='black', linestyle='--', linewidth=1, label=f'Avg: {avg_vol:.1f}%')
        ax.legend()
    
    def _plot_rolling_var(self, ax):
        """Plot rolling VaR and CVaR."""
        # Calculate rolling VaR and CVaR
        window = min(252, len(self.results_df) // 4)
        if window > 20:  # Only calculate if we have enough data
            # Calculate rolling 95% VaR
            self.results_df['rolling_var95'] = self.results_df['return'].rolling(window).quantile(0.05) * 100
            
            # Calculate rolling 95% CVaR (Expected Shortfall)
            def rolling_cvar(x):
                return x[x <= np.quantile(x, 0.05)].mean() * 100
            
            self.results_df['rolling_cvar95'] = self.results_df['return'].rolling(window).apply(
                rolling_cvar, raw=True
            )
            
            # Plot rolling VaR and CVaR
            self.results_df['rolling_var95'].plot(ax=ax, color='#0066CC', linewidth=1.5, label='VaR (95%)')
            self.results_df['rolling_cvar95'].plot(ax=ax, color='#CC0000', linewidth=1.5, label='CVaR (95%)')
            
            # Format the axis
            ax.set_title('Rolling VaR & CVaR (252-day)', fontsize=12)
            ax.set_ylabel('Daily Loss (%)')
            
            # Format x-axis with dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
        else:
            # Not enough data
            ax.text(0.5, 0.5, 'Insufficient data for rolling VaR/CVaR', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rolling VaR & CVaR', fontsize=12)
    
    def _plot_drawdown_periods(self, ax):
        """Plot drawdown underwater chart with highlighted periods."""
        # Plot the drawdown underwater chart
        self.results_df['drawdown'].mul(100).plot(ax=ax, color='#CC0000', linewidth=1.5, alpha=0.7)
        
        # Fill the area
        ax.fill_between(self.results_df.index, 0, self.results_df['drawdown'].mul(100), color='#CC0000', alpha=0.3)
        
        # Format the axis
        ax.set_title('Drawdown Periods', fontsize=12)
        ax.set_ylabel('Drawdown (%)')
        
        # Format y-axis as percentage and invert
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.invert_yaxis()
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Identify major drawdown periods (e.g., drawdowns > 10%)
        # First, identify where drawdowns cross below -0.1 (-10%)
        threshold = -0.1
        crosses_below = (self.results_df['drawdown'] < threshold) & (self.results_df['drawdown'].shift(1) >= threshold)
        crosses_above = (self.results_df['drawdown'] >= threshold) & (self.results_df['drawdown'].shift(1) < threshold)
        
        # Find start and end dates of major drawdowns
        start_dates = self.results_df.index[crosses_below]
        end_dates = self.results_df.index[crosses_above]
        
        # Adjust if we start in a drawdown or end in a drawdown
        if sum(self.results_df['drawdown'] < threshold) > 0:  # Check if we have any major drawdowns
            if not any(crosses_below) or self.results_df['drawdown'].iloc[0] < threshold:
                start_dates = start_dates.insert(0, self.results_df.index[0])
            if not any(crosses_above) or self.results_df['drawdown'].iloc[-1] < threshold:
                end_dates = end_dates.append(pd.Index([self.results_df.index[-1]]))
        
        # Highlight major drawdown periods
        for i in range(min(len(start_dates), len(end_dates))):
            if i < len(start_dates) and i < len(end_dates):
                ax.axvspan(start_dates[i], end_dates[i], color='red', alpha=0.2)
    
    def _plot_return_distribution(self, ax):
        """Plot return distribution with normal curve for comparison."""
        # Get return data
        returns = self.results_df['return'].dropna() * 100
        
        # Plot histogram
        sns.histplot(returns, ax=ax, bins=50, kde=True, color='#0066CC', stat='density', alpha=0.7)
        
        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        mean = returns.mean()
        std = returns.std()
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r--', linewidth=1.5, label='Normal Dist.')
        
        # Format the axis
        ax.set_title('Return Distribution vs. Normal', fontsize=12)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Density')
        
        # Add statistics annotations
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        textstr = '\n'.join((
            f'Mean: {mean:.2f}%',
            f'Std Dev: {std:.2f}%',
            f'Skew: {stats.skew(returns):.2f}',
            f'Kurt: {stats.kurtosis(returns):.2f}'
        ))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def create_trade_analysis(self, save_path=None):
        """
        Create trade analysis visualizations.
        
        Args:
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Identify weight columns
        weight_cols = [col for col in self.results_df.columns if col.startswith('weight_')]
        
        if not weight_cols:
            logger.warning("No position data found in results DataFrame")
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No position data available', ha='center', va='center')
            return fig
        
        # Create figure with gridspec - FIXED: use constrained_layout instead of tight_layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Title the figure
        fig.suptitle('Trade Analysis Dashboard', fontsize=16, y=0.98)
        
        # 1. Position changes over time (turnover)
        ax_turnover = fig.add_subplot(gs[0, 0])
        self._plot_turnover(ax_turnover, weight_cols)
        
        # 2. Position sizing distribution
        ax_sizing = fig.add_subplot(gs[0, 1])
        self._plot_position_sizing(ax_sizing, weight_cols)
        
        # 3. Trade entry/exit points on performance
        ax_trades = fig.add_subplot(gs[1, :])
        self._plot_major_trades(ax_trades, weight_cols)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Trade analysis saved to {save_path}")
        
        return fig
    
    def _plot_turnover(self, ax, weight_cols):
        """Plot portfolio turnover over time."""
        # Calculate daily position changes (turnover)
        position_changes = self.results_df[weight_cols].diff().abs().sum(axis=1)
        
        # Calculate rolling 20-day turnover
        rolling_turnover = position_changes.rolling(20).sum()
        
        # Plot turnover
        rolling_turnover.mul(100).plot(ax=ax, color='#0066CC', linewidth=1.5)
        
        # Format the axis
        ax.set_title('20-Day Rolling Turnover', fontsize=12)
        ax.set_ylabel('Turnover (%)')
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add average line
        avg_turnover = rolling_turnover.mean() * 100
        ax.axhline(avg_turnover, color='black', linestyle='--', linewidth=1, label=f'Avg: {avg_turnover:.1f}%')
        ax.legend()
    
    def _plot_position_sizing(self, ax, weight_cols):
        """Plot position sizing distribution."""
        # Get all non-zero weights
        all_weights = self.results_df[weight_cols].values.flatten()
        all_weights = all_weights[all_weights != 0]
        
        # Plot histogram of position sizes
        sns.histplot(all_weights * 100, ax=ax, bins=50, kde=True, color='#0066CC')
        
        # Format the axis
        ax.set_title('Position Size Distribution', fontsize=12)
        ax.set_xlabel('Position Size (%)')
        ax.set_ylabel('Frequency')
        
        # Add statistics annotations
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        textstr = '\n'.join((
            f'Mean: {np.mean(all_weights)*100:.2f}%',
            f'Median: {np.median(all_weights)*100:.2f}%',
            f'Min: {np.min(all_weights)*100:.2f}%',
            f'Max: {np.max(all_weights)*100:.2f}%'
        ))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Add vertical lines for mean and median
        ax.axvline(np.mean(all_weights) * 100, color='red', linestyle='-', linewidth=1, label='Mean')
        ax.axvline(np.median(all_weights) * 100, color='green', linestyle='--', linewidth=1, label='Median')
        ax.legend()
    
    def _plot_major_trades(self, ax, weight_cols):
        """Plot major trades entry/exit on performance chart."""
        # Plot cumulative return
        self.results_df['cumulative_return'].mul(100).plot(ax=ax, color='#0066CC', linewidth=1.5)
        
        # Identify significant position changes
        # Look for large changes in any position
        position_changes = self.results_df[weight_cols].diff().abs()
        
        # Get top 10 largest position changes
        large_changes = position_changes.max(axis=1).nlargest(10)
        
        # For each large change, identify which asset had the largest change
        significant_trades = []
        
        for date in large_changes.index:
            changes = position_changes.loc[date]
            max_change_asset = changes.idxmax()
            asset_name = max_change_asset.replace('weight_', '')
            
            # Get direction of change
            direction = 'Buy' if self.results_df[max_change_asset].diff().loc[date] > 0 else 'Sell'
            
            # Get cumulative return at this point
            cum_ret = self.results_df['cumulative_return'].loc[date] * 100
            
            significant_trades.append((date, asset_name, direction, cum_ret))
        
        # Plot markers for significant trades
        for date, asset, direction, cum_ret in significant_trades:
            color = 'green' if direction == 'Buy' else 'red'
            marker = '^' if direction == 'Buy' else 'v'
            
            ax.scatter(date, cum_ret, color=color, marker=marker, s=100, zorder=5)
            ax.annotate(f'{direction} {asset}', 
                       xy=(date, cum_ret),
                       xytext=(10, 10 if direction == 'Buy' else -15), 
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', color='black'),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Format the axis
        ax.set_title('Cumulative Return with Major Trades', fontsize=12)
        ax.set_ylabel('Return (%)')
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend for trade markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Buy'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Sell')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    
    def save_all_visualizations(self, output_dir):
        """
        Save all visualizations to the specified output directory.
        
        Args:
            output_dir (str): Directory to save visualizations
            
        Returns:
            list: List of saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Performance dashboard
        perf_path = os.path.join(output_dir, f'performance_dashboard_{timestamp}.png')
        self.create_performance_dashboard(save_path=perf_path)
        saved_paths.append(perf_path)
        
        # Position analysis
        pos_path = os.path.join(output_dir, f'position_analysis_{timestamp}.png')
        self.create_position_analysis(save_path=pos_path)
        saved_paths.append(pos_path)
        
        # Risk analysis
        risk_path = os.path.join(output_dir, f'risk_analysis_{timestamp}.png')
        self.create_risk_analysis(save_path=risk_path)
        saved_paths.append(risk_path)
        
        # Trade analysis
        trade_path = os.path.join(output_dir, f'trade_analysis_{timestamp}.png')
        self.create_trade_analysis(save_path=trade_path)
        saved_paths.append(trade_path)
        
        logger.info(f"All visualizations saved to {output_dir}")
        return saved_paths