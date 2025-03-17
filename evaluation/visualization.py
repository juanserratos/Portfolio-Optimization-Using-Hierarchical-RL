"""
Enhanced visualization module for the deep RL trading framework using Plotly.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import os
import logging
from scipy import stats
import calendar

# Configure logging
logger = logging.getLogger(__name__)

# Set default plotly template
pio.templates.default = "plotly_white"

class BacktestVisualizer:
    """Class for creating enhanced visualizations of backtest results using Plotly."""
    
    def __init__(self, results_df, benchmark_returns=None, figsize=(900, 600), theme="plotly_white"):
        """
        Initialize the backtest visualizer.
        
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
        
    def create_performance_dashboard(self, show_benchmark=True, show_rolling_metrics=True, save_path=None, show=True):
        """
        Create a comprehensive performance dashboard.
        
        Args:
            show_benchmark (bool): Whether to show benchmark comparison
            show_rolling_metrics (bool): Whether to show rolling metrics
            save_path (str, optional): Path to save the figure
            show (bool): Whether to display the figure
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.5, 0.25, 0.25],
            column_widths=[1.0, 1.0],
            subplot_titles=(
                'Cumulative Returns', '',
                'Drawdowns', '',
                'Monthly Returns', 'Return Distribution'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"type": "heatmap"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.10,
            horizontal_spacing=0.05
        )

        # Add main title
        fig.update_layout(
            title={
                'text': 'Trading Strategy Performance Dashboard',
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
        
        # 1. Cumulative returns plot
        self._add_cumulative_returns(fig, row=1, col=1, show_benchmark=show_benchmark)
        
        # 2. Drawdown plot
        self._add_drawdowns(fig, row=2, col=1)
        
        # 3. Monthly returns heatmap
        self._add_monthly_returns_heatmap(fig, row=3, col=1)
        
        # 4. Return histogram
        self._add_return_histogram(fig, row=3, col=2)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Performance dashboard saved to {save_path}")
        
        # Show plot if requested
        if show:
            fig.show()
            
        return fig
    
    def _add_cumulative_returns(self, fig, row=1, col=1, show_benchmark=True):
        """Add cumulative returns plot to the figure."""
        # Plot strategy returns
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=self.results_df['cumulative_return'] * 100,
                mode='lines',
                name='Strategy',
                line=dict(color='#0066CC', width=2)
            ),
            row=row, col=col
        )
        
        # Add benchmark if available
        if show_benchmark and self.benchmark_returns is not None:
            benchmark_cumret = self.benchmark_cumulative - 1
            # Align benchmark to the same time period
            benchmark_cumret = benchmark_cumret.loc[benchmark_cumret.index.intersection(self.results_df.index)]
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumret.index,
                    y=benchmark_cumret * 100,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#999999', width=1.5, dash='dash')
                ),
                row=row, col=col
            )
        
        # Format axes
        fig.update_yaxes(
            title_text='Return (%)',
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='black',
            row=row, col=col
        )
        
        # Add final return annotation
        final_return = self.results_df['cumulative_return'].iloc[-1] * 100
        fig.add_annotation(
            x=self.results_df.index[-1],
            y=final_return,
            text=f"{final_return:.2f}%",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(size=12, color="black", family="Arial"),
            bgcolor="white",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            row=row, col=col
        )
    
    def _add_drawdowns(self, fig, row=2, col=1):
        """Add drawdowns plot to the figure."""
        # Plot drawdowns
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=self.results_df['drawdown'] * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='#CC0000', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(204,0,0,0.1)'
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Drawdown (%)',
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='black',
            autorange="reversed",  # Invert y-axis to show drawdowns as negative
            row=row, col=col
        )
        
        # Highlight maximum drawdown
        max_drawdown_idx = self.results_df['drawdown'].idxmin()
        max_drawdown = self.results_df.loc[max_drawdown_idx, 'drawdown'] * 100
        
        fig.add_annotation(
            x=max_drawdown_idx,
            y=max_drawdown,
            text=f"Max DD: {max_drawdown:.2f}%",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=0,
            font=dict(size=12, color="#CC0000", family="Arial"),
            bgcolor="white",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            row=row, col=col
        )
    
    def _add_monthly_returns_heatmap(self, fig, row=3, col=1):
        """Add monthly returns heatmap to the figure."""
        if self.monthly_returns.empty:
            fig.add_annotation(
                x=0.5, y=0.5,
                text='Insufficient data for monthly heatmap',
                showarrow=False,
                font=dict(size=12),
                row=row, col=col
            )
            return
        
        # Format month names and handle potentially missing data
        month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
        formatted_data = self.monthly_returns.rename(columns=month_names)
        
        # Convert to format suitable for Plotly heatmap
        z_data = formatted_data.values * 100  # Convert to percentage
        x_data = [month_names[i] for i in formatted_data.columns]
        y_data = formatted_data.index.get_level_values(0)  # Years
        
        # Create custom text for hover
        hover_text = [[f"{z_data[i][j]:.2f}%" for j in range(len(x_data))] for i in range(len(y_data))]
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_data,
                y=y_data,
                colorscale='RdYlGn',
                zmid=0,
                text=hover_text,
                hoverinfo='text+x+y',
                colorbar=dict(
                    title="Return (%)",
                    thickness=10,
                    len=0.6,
                    y=0.8,
                    x=1.1
                )
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Year',
            row=row, col=col
        )
    
    def _add_return_histogram(self, fig, row=3, col=2):
        """Add return distribution histogram to the figure."""
        # Calculate statistics for annotations
        returns = self.results_df['return'] * 100
        mean_return = returns.mean()
        median_return = returns.median()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                opacity=0.7,
                name='Daily Returns',
                marker_color='#0066CC',
                histnorm='probability density'
            ),
            row=row, col=col
        )
        
        # Add KDE (kernel density estimate)
        x_kde = np.linspace(returns.min(), returns.max(), 100)
        kde = stats.gaussian_kde(returns.dropna())
        y_kde = kde(x_kde)
        
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode='lines',
                name='Density',
                line=dict(color='#CC0000', width=2)
            ),
            row=row, col=col
        )
        
        # Add mean and median lines
        fig.add_trace(
            go.Scatter(
                x=[mean_return, mean_return],
                y=[0, max(y_kde) * 1.1],
                mode='lines',
                name=f'Mean: {mean_return:.2f}%',
                line=dict(color='#CC0000', width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=[median_return, median_return],
                y=[0, max(y_kde) * 1.1],
                mode='lines',
                name=f'Median: {median_return:.2f}%',
                line=dict(color='green', width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_xaxes(
            title_text='Daily Return (%)',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Density',
            row=row, col=col
        )
    
    def create_position_analysis(self, num_assets=10, save_path=None, show=True):
        """
        Create position and allocation analysis plots.
        
        Args:
            num_assets (int): Number of top assets to display
            save_path (str, optional): Path to save the figure
            show (bool): Whether to display the figure
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Identify weight columns
        weight_cols = [col for col in self.results_df.columns if col.startswith('weight_')]
        
        if not weight_cols:
            logger.warning("No position data found in results DataFrame")
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text='No position data available',
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Asset Allocation Over Time',
                'Current Allocation',
                'Top Asset Allocations',
                'Allocation Heatmap (Monthly)'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "domain"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.15
        )

        # Add main title
        fig.update_layout(
            title={
                'text': 'Portfolio Allocation Analysis',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            height=self.figsize[1],
            width=self.figsize[0]
        )
        
        # 1. Asset allocation over time
        self._add_asset_allocation(fig, weight_cols, num_assets, row=1, col=1)
        
        # 2. Current allocation pie chart
        self._add_current_allocation(fig, weight_cols, num_assets, row=2, col=1)
        
        # 3. Allocation heatmap
        self._add_allocation_heatmap(fig, weight_cols, num_assets, row=2, col=2)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Position analysis saved to {save_path}")
        
        # Show if requested
        if show:
            fig.show()
            
        return fig
    
    def _add_asset_allocation(self, fig, weight_cols, num_assets=10, row=1, col=1):
        """Add asset allocation over time to the figure."""
        # Get top assets by average absolute weight
        top_assets = self._get_top_assets(weight_cols, n=num_assets)
        
        # Extract data
        allocation_data = self.results_df[top_assets].abs() * 100  # Convert to percentage
        
        # Plot each asset allocation as a line
        for asset in top_assets:
            ticker = asset.replace('weight_', '')
            fig.add_trace(
                go.Scatter(
                    x=self.results_df.index,
                    y=allocation_data[asset],
                    mode='lines',
                    name=ticker,
                    stackgroup='one'  # Stacked area chart
                ),
                row=row, col=col
            )
        
        # Format axes
        fig.update_yaxes(
            title_text='Allocation (%)',
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, 100],
            row=row, col=col
        )
    
    def _add_current_allocation(self, fig, weight_cols, num_assets=10, row=2, col=1):
        """Add current allocation pie chart to the figure."""
        # Get the last row of data
        current_weights = self.results_df[weight_cols].iloc[-1]
        
        # Get the top assets
        top_weights = current_weights.abs().nlargest(num_assets)
        top_actual_weights = current_weights[top_weights.index]
        
        # Convert ticker names for display
        labels = [col.replace('weight_', '') for col in top_actual_weights.index]
        values = top_actual_weights.abs() * 100  # Convert to percentage
        
        # Add pie chart
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo='label+percent',
                insidetextorientation='radial',
                marker=dict(
                    line=dict(color='#FFFFFF', width=1)
                ),
                textfont=dict(size=12),
            ),
            row=row, col=col
        )
    
    def _add_allocation_heatmap(self, fig, weight_cols, num_assets=10, row=2, col=2):
        """Add allocation heatmap to the figure."""
        # Get top assets
        top_assets = self._get_top_assets(weight_cols, n=num_assets)
        
        # Resample to monthly data to reduce density
        monthly_data = self.results_df[top_assets].resample('ME').last() * 100  # Convert to percentage
        
        # Create labels for axes
        date_labels = [d.strftime('%Y-%m') for d in monthly_data.index]
        ticker_labels = [col.replace('weight_', '') for col in monthly_data.columns]
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=monthly_data.T.values,
                x=date_labels,
                y=ticker_labels,
                colorscale='RdBu_r',
                zmid=0,  # Center colorscale at 0
                colorbar=dict(
                    title="Allocation (%)",
                    thickness=10,
                    len=0.6
                )
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Asset',
            row=row, col=col
        )
        
        fig.update_xaxes(
            title_text='Date',
            tickangle=45,
            row=row, col=col
        )
    
    def _get_top_assets(self, weight_cols, n=10):
        """Get the top n assets by average absolute weight."""
        avg_weights = self.results_df[weight_cols].abs().mean().nlargest(n)
        return avg_weights.index.tolist()
    
    def create_risk_analysis(self, save_path=None, show=True):
        """
        Create risk analysis visualizations.
        
        Args:
            save_path (str, optional): Path to save the figure
            show (bool): Whether to display the figure
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rolling Volatility (252-day)',
                'Rolling VaR & CVaR (252-day)',
                'Drawdown Periods',
                'Return Distribution vs. Normal'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.10
        )

        # Add main title
        fig.update_layout(
            title={
                'text': 'Risk Analysis Dashboard',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            height=self.figsize[1],
            width=self.figsize[0]
        )
        
        # 1. Rolling volatility
        self._add_rolling_volatility(fig, row=1, col=1)
        
        # 2. Rolling VaR and CVaR
        self._add_rolling_var(fig, row=1, col=2)
        
        # 3. Drawdown periods
        self._add_drawdown_periods(fig, row=2, col=1)
        
        # 4. Return distribution
        self._add_return_distribution_with_normal(fig, row=2, col=2)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Risk analysis saved to {save_path}")
            
        # Show if requested
        if show:
            fig.show()
        
        return fig
    
    def _add_rolling_volatility(self, fig, row=1, col=1):
        """Add rolling volatility plot to the figure."""
        # Calculate rolling volatility if not already done
        if 'rolling_volatility' not in self.results_df.columns:
            window = min(252, len(self.results_df) // 4)
            self.results_df['rolling_volatility'] = self.results_df['return'].rolling(window).std() * np.sqrt(252)
        
        # Plot rolling volatility
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=self.results_df['rolling_volatility'] * 100,
                mode='lines',
                name='Rolling Vol',
                line=dict(color='#CC0000', width=2)
            ),
            row=row, col=col
        )
        
        # Add average line
        avg_vol = self.results_df['rolling_volatility'].mean() * 100
        fig.add_trace(
            go.Scatter(
                x=[self.results_df.index[0], self.results_df.index[-1]],
                y=[avg_vol, avg_vol],
                mode='lines',
                name=f'Avg: {avg_vol:.1f}%',
                line=dict(color='black', width=1, dash='dash')
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Annualized Volatility (%)',
            gridcolor='rgba(0,0,0,0.1)',
            row=row, col=col
        )
    
    def _add_rolling_var(self, fig, row=1, col=1):
        """Add rolling VaR and CVaR plot to the figure."""
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
            
            # Plot rolling VaR
            fig.add_trace(
                go.Scatter(
                    x=self.results_df.index,
                    y=self.results_df['rolling_var95'],
                    mode='lines',
                    name='VaR (95%)',
                    line=dict(color='#0066CC', width=2)
                ),
                row=row, col=col
            )
            
            # Plot rolling CVaR
            fig.add_trace(
                go.Scatter(
                    x=self.results_df.index,
                    y=self.results_df['rolling_cvar95'],
                    mode='lines',
                    name='CVaR (95%)',
                    line=dict(color='#CC0000', width=2)
                ),
                row=row, col=col
            )
            
            # Format axes
            fig.update_yaxes(
                title_text='Daily Loss (%)',
                gridcolor='rgba(0,0,0,0.1)',
                row=row, col=col
            )
        else:
            # Not enough data
            fig.add_annotation(
                x=0.5, y=0.5,
                text='Insufficient data for rolling VaR/CVaR',
                showarrow=False,
                font=dict(size=12),
                row=row, col=col
            )
    
    def _add_drawdown_periods(self, fig, row=1, col=1):
        """Add drawdown periods plot to the figure."""
        # Plot the drawdown underwater chart
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=self.results_df['drawdown'] * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='#CC0000', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(204,0,0,0.3)'
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Drawdown (%)',
            gridcolor='rgba(0,0,0,0.1)',
            autorange="reversed",  # Invert y-axis
            row=row, col=col
        )
        
        # Identify major drawdown periods (e.g., drawdowns > 10%)
        threshold = -0.1
        crosses_below = (self.results_df['drawdown'] < threshold) & (self.results_df['drawdown'].shift(1) >= threshold)
        crosses_above = (self.results_df['drawdown'] >= threshold) & (self.results_df['drawdown'].shift(1) < threshold)
        
        # Find start and end dates of major drawdowns
        start_dates = self.results_df.index[crosses_below]
        end_dates = self.results_df.index[crosses_above]
        
        # Adjust if we start in a drawdown or end in a drawdown
        if sum(self.results_df['drawdown'] < threshold) > 0:  # Check if we have any major drawdowns
            if not any(crosses_below) or self.results_df['drawdown'].iloc[0] < threshold:
                start_dates = pd.Index([self.results_df.index[0]]).append(start_dates)
            if not any(crosses_above) or self.results_df['drawdown'].iloc[-1] < threshold:
                end_dates = end_dates.append(pd.Index([self.results_df.index[-1]]))
        
        # Add rectangles for major drawdown periods
        for i in range(min(len(start_dates), len(end_dates))):
            if i < len(start_dates) and i < len(end_dates):
                fig.add_vrect(
                    x0=start_dates[i],
                    x1=end_dates[i],
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    row=row, col=col
                )
    
    def _add_return_distribution_with_normal(self, fig, row=1, col=1):
        """Add return distribution with normal curve to the figure."""
        # Get return data
        returns = self.results_df['return'].dropna() * 100
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                opacity=0.7,
                name='Returns',
                marker_color='#0066CC',
                histnorm='probability density'
            ),
            row=row, col=col
        )
        
        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        mean = returns.mean()
        std = returns.std()
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
        
        # Add statistics annotation
        stats_text = (
            f'Mean: {mean:.2f}%<br>'
            f'Std Dev: {std:.2f}%<br>'
            f'Skew: {stats.skew(returns):.2f}<br>'
            f'Kurt: {stats.kurtosis(returns):.2f}'
        )
        
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="x domain",
            yref="y domain",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            align="left",
            row=row, col=col
        )
        
        # Format axes
        fig.update_xaxes(
            title_text='Daily Return (%)',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Density',
            row=row, col=col
        )
    
    def create_trade_analysis(self, save_path=None, show=True):
        """
        Create trade analysis visualizations.
        
        Args:
            save_path (str, optional): Path to save the figure
            show (bool): Whether to display the figure
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Identify weight columns
        weight_cols = [col for col in self.results_df.columns if col.startswith('weight_')]
        
        if not weight_cols:
            logger.warning("No position data found in results DataFrame")
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text='No position data available',
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '20-Day Rolling Turnover',
                'Position Size Distribution',
                'Cumulative Return with Major Trades',
                ''
            ),
            specs=[
                [{}, {}],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.15
        )

        # Add main title
        fig.update_layout(
            title={
                'text': 'Trade Analysis Dashboard',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            height=self.figsize[1],
            width=self.figsize[0]
        )
        
        # 1. Position changes over time (turnover)
        self._add_turnover(fig, weight_cols, row=1, col=1)
        
        # 2. Position sizing distribution
        self._add_position_sizing(fig, weight_cols, row=1, col=2)
        
        # 3. Trade entry/exit points on performance
        self._add_major_trades(fig, weight_cols, row=2, col=1)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Trade analysis saved to {save_path}")
            
        # Show if requested
        if show:
            fig.show()
        
        return fig
    
    def _add_turnover(self, fig, weight_cols, row=1, col=1):
        """Add portfolio turnover plot to the figure."""
        # Calculate daily position changes (turnover)
        position_changes = self.results_df[weight_cols].diff().abs().sum(axis=1)
        
        # Calculate rolling 20-day turnover
        rolling_turnover = position_changes.rolling(20).sum() * 100  # Convert to percentage
        
        # Plot turnover
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=rolling_turnover,
                mode='lines',
                name='Turnover',
                line=dict(color='#0066CC', width=2)
            ),
            row=row, col=col
        )
        
        # Add average line
        avg_turnover = rolling_turnover.mean()
        fig.add_trace(
            go.Scatter(
                x=[self.results_df.index[0], self.results_df.index[-1]],
                y=[avg_turnover, avg_turnover],
                mode='lines',
                name=f'Avg: {avg_turnover:.1f}%',
                line=dict(color='black', width=1, dash='dash')
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(
            title_text='Turnover (%)',
            gridcolor='rgba(0,0,0,0.1)',
            row=row, col=col
        )
    
    def _add_position_sizing(self, fig, weight_cols, row=1, col=1):
        """Add position sizing distribution to the figure."""
        # Get all non-zero weights
        all_weights = self.results_df[weight_cols].values.flatten()
        all_weights = all_weights[all_weights != 0] * 100  # Convert to percentage
        
        # Calculate statistics
        mean_weight = np.mean(all_weights)
        median_weight = np.median(all_weights)
        min_weight = np.min(all_weights)
        max_weight = np.max(all_weights)
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=all_weights,
                nbinsx=50,
                opacity=0.7,
                name='Position Size',
                marker_color='#0066CC'
            ),
            row=row, col=col
        )
        
        # Add vertical lines for mean and median
        fig.add_trace(
            go.Scatter(
                x=[mean_weight, mean_weight],
                y=[0, 1],
                mode='lines',
                name=f'Mean: {mean_weight:.2f}%',
                line=dict(color='red', width=2)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=[median_weight, median_weight],
                y=[0, 1],
                mode='lines',
                name=f'Median: {median_weight:.2f}%',
                line=dict(color='green', width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        # Add statistics annotation
        stats_text = (
            f'Mean: {mean_weight:.2f}%<br>'
            f'Median: {median_weight:.2f}%<br>'
            f'Min: {min_weight:.2f}%<br>'
            f'Max: {max_weight:.2f}%'
        )
        
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="x domain",
            yref="y domain",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            align="left",
            row=row, col=col
        )
        
        # Format axes
        fig.update_xaxes(
            title_text='Position Size (%)',
            row=row, col=col
        )
        
        fig.update_yaxes(
            title_text='Frequency',
            row=row, col=col
        )
        
        # Update y-axis range after plotting
        fig.update_yaxes(
            autorange=True,
            row=row, col=col
        )
    
    def _add_major_trades(self, fig, weight_cols, row=1, col=1):
        """Add major trades entry/exit on performance chart to the figure."""
        # Plot cumulative return
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=self.results_df['cumulative_return'] * 100,
                mode='lines',
                name='Strategy',
                line=dict(color='#0066CC', width=2)
            ),
            row=row, col=col
        )
        
        # Identify significant position changes
        # Look for large changes in any position
        position_changes = self.results_df[weight_cols].diff().abs()
        
        # Get top 10 largest position changes
        large_changes = position_changes.max(axis=1).nlargest(10)
        
        # For each large change, identify which asset had the largest change
        buy_dates = []
        buy_assets = []
        buy_returns = []
        sell_dates = []
        sell_assets = []
        sell_returns = []
        
        for date in large_changes.index:
            changes = position_changes.loc[date]
            max_change_asset = changes.idxmax()
            asset_name = max_change_asset.replace('weight_', '')
            
            # Get direction of change
            direction = 'Buy' if self.results_df[max_change_asset].diff().loc[date] > 0 else 'Sell'
            
            # Get cumulative return at this point
            cum_ret = self.results_df['cumulative_return'].loc[date] * 100
            
            if direction == 'Buy':
                buy_dates.append(date)
                buy_assets.append(asset_name)
                buy_returns.append(cum_ret)
            else:
                sell_dates.append(date)
                sell_assets.append(asset_name)
                sell_returns.append(cum_ret)
        
        # Add buy markers
        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_returns,
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(color='black', width=1)
                    ),
                    text=[f'Buy {asset}' for asset in buy_assets],
                    hovertemplate='%{text}<br>Date: %{x}<br>Return: %{y:.2f}%'
                ),
                row=row, col=col
            )
        
        # Add sell markers
        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_returns,
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(color='black', width=1)
                    ),
                    text=[f'Sell {asset}' for asset in sell_assets],
                    hovertemplate='%{text}<br>Date: %{x}<br>Return: %{y:.2f}%'
                ),
                row=row, col=col
            )
        
        # Format axes
        fig.update_yaxes(
            title_text='Return (%)',
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='black',
            row=row, col=col
        )
    
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
        self.create_performance_dashboard(save_path=perf_path, show=False)
        saved_paths.append(perf_path)
        
        # Position analysis
        pos_path = os.path.join(output_dir, f'position_analysis_{timestamp}.png')
        self.create_position_analysis(save_path=pos_path, show=False)
        saved_paths.append(pos_path)
        
        # Risk analysis
        risk_path = os.path.join(output_dir, f'risk_analysis_{timestamp}.png')
        self.create_risk_analysis(save_path=risk_path, show=False)
        saved_paths.append(risk_path)
        
        # Trade analysis
        trade_path = os.path.join(output_dir, f'trade_analysis_{timestamp}.png')
        self.create_trade_analysis(save_path=trade_path, show=False)
        saved_paths.append(trade_path)
        
        # Also save interactive HTML versions
        for path in saved_paths:
            html_path = path.replace('.png', '.html')
            if 'performance' in path:
                fig = self.create_performance_dashboard(show=False)
            elif 'position' in path:
                fig = self.create_position_analysis(show=False)
            elif 'risk' in path:
                fig = self.create_risk_analysis(show=False)
            elif 'trade' in path:
                fig = self.create_trade_analysis(show=False)
            
            # Save as HTML
            fig.write_html(html_path)
            saved_paths.append(html_path)
        
        logger.info(f"All visualizations saved to {output_dir}")
        return saved_paths