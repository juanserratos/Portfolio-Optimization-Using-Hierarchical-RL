"""
Updates to the backtest engine with simplified visualization options.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm
from datetime import datetime
from utils.metrics import calculate_performance_metrics
from evaluation.visualization import BacktestVisualizer
from evaluation.portfolio_analytics import PortfolioAnalytics

# Configure logging
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Enhanced engine for backtesting trading strategies with advanced visualizations
    and portfolio analytics.
    """
    
    def __init__(self, env, model, device=None):
        """
        Initialize backtest engine.
        
        Args:
            env: Trading environment
            model: Trained model
            device: Torch device (cpu or cuda)
        """
        self.env = env
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize history storage
        self.reset_history()
        
        # Initialize visualizer and analytics to None (will be created after running backtest)
        self.visualizer = None
        self.analytics = None
        
    def reset_history(self):
        """Reset history storage."""
        self.portfolio_values = []
        self.returns = []
        self.positions = []
        self.transactions = []
        self.dates = []
        
    def run_backtest(self, deterministic=True, progress_bar=True):
        """
        Run backtest.
        
        Args:
            deterministic (bool): Whether to use deterministic actions
            progress_bar (bool): Whether to show progress bar
            
        Returns:
            dict: Performance metrics
        """
        logger.info("Starting backtest")
        
        # Reset environment and history
        obs = self.env.reset()
        self.reset_history()
        self.portfolio_values.append(1.0)  # Start with initial value
        
        # Store dates if available
        if hasattr(self.env, 'price_data') and hasattr(self.env.price_data, 'index'):
            start_idx = self.env.window_size
            if start_idx < len(self.env.price_data.index):
                self.dates.append(self.env.price_data.index[start_idx])
            else:
                # Fallback if start_idx is out of range
                self.dates.append(pd.Timestamp('2000-01-01'))
        else:
            self.dates.append(pd.Timestamp('2000-01-01'))
        
        # Reset tracking variables
        done = False
        
        # Create progress bar if requested
        if progress_bar:
            if hasattr(self.env, 'price_data'):
                total_steps = len(self.env.price_data) - self.env.window_size
            else:
                total_steps = 1000  # Default if we don't know the length
            pbar = tqdm(total=total_steps, desc="Running backtest")
        
        # Run through entire environment
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from model
            with torch.no_grad():
                action, _, _ = self.model.get_action(obs_tensor, deterministic=deterministic)
                
            # If action is a dict (hierarchical), convert it to numpy
            if isinstance(action, dict):
                action_np = {
                    'strategic': action['strategic'].squeeze().cpu().numpy(),
                    'tactical': {
                        class_name: action['tactical'][class_name].squeeze().cpu().numpy()
                        for class_name in action['tactical']
                    }
                }
            else:
                action_np = action.squeeze().cpu().numpy()
            
            # Take action in environment
            next_obs, reward, done, info = self.env.step(action_np)
            
            # Store history
            self.portfolio_values.append(info['portfolio_value'])
            self.positions.append(self.env.current_weights.copy())
            
            # Store date if available
            if hasattr(self.env, 'price_data') and hasattr(self.env.price_data, 'index'):
                current_step = self.env.current_step
                if current_step < len(self.env.price_data.index):
                    self.dates.append(self.env.price_data.index[current_step])
                else:
                    # Fallback if current_step is out of range
                    self.dates.append(self.dates[-1] + pd.Timedelta(days=1))
            else:
                self.dates.append(self.dates[-1] + pd.Timedelta(days=1))
            
            # Calculate transactions (change in positions)
            if len(self.positions) > 1:
                transaction = np.abs(self.positions[-1] - self.positions[-2])
                self.transactions.append(transaction)
            
            # Store returns
            if len(self.portfolio_values) > 1:
                ret = self.portfolio_values[-1] / self.portfolio_values[-2] - 1
                self.returns.append(ret)
            
            # Update observation
            obs = next_obs
            
            # Update progress bar
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix({
                    'portfolio_value': f'{self.portfolio_values[-1]:.2f}',
                    'return': f'{self.returns[-1] * 100 if self.returns else 0:.2f}%'
                })
        
        # Close progress bar
        if progress_bar:
            pbar.close()
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(
            returns=self.returns,
            portfolio_values=self.portfolio_values
        )
        
        logger.info(f"Backtest completed with final portfolio value: {self.portfolio_values[-1]:.4f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}, Max Drawdown: {metrics['max_drawdown']:.4f}")
        
        # Create results dataframe for the visualizer
        self._create_results_dataframe()
        
        return metrics
    
    def _create_results_dataframe(self):
        """Create a results dataframe with all necessary data for visualization."""
        # Create DataFrame with dates from environment
        results = pd.DataFrame({
            'portfolio_value': self.portfolio_values,
        }, index=self.dates)
        
        # Add returns
        results['return'] = results['portfolio_value'].pct_change().fillna(0)
        
        # Fix for length mismatch - ensure positions array has the right dimensions
        positions_array = np.array(self.positions)
        
        # Check for length mismatch and fix if necessary
        if len(positions_array) != len(results):
            logger.warning(f"Length mismatch: positions ({len(positions_array)}) vs results ({len(results)})")
            
            # If positions array is shorter, pad with zeros or the last position
            if len(positions_array) < len(results):
                # Create a padding array with the same number of assets
                padding_shape = (len(results) - len(positions_array), positions_array.shape[1])
                # Use the last position as padding
                padding = np.tile(positions_array[-1], (len(results) - len(positions_array), 1))
                positions_array = np.vstack([positions_array, padding])
            # If positions array is longer, truncate it
            else:
                positions_array = positions_array[:len(results)]
        
        # Get asset names from environment
        if hasattr(self.env, 'price_data') and hasattr(self.env.price_data, 'columns'):
            asset_names = self.env.price_data.columns
        else:
            asset_names = [f'Asset {i+1}' for i in range(positions_array.shape[1])]
        
        # Add positions for each asset
        for i, asset in enumerate(asset_names):
            if i < positions_array.shape[1]:
                results[f'weight_{asset}'] = positions_array[:, i]
        
        # Add cumulative return and drawdown
        results['cumulative_return'] = results['portfolio_value'] / results['portfolio_value'].iloc[0] - 1
        running_max = results['cumulative_return'].cummax()
        results['drawdown'] = (results['cumulative_return'] - running_max) / (1 + running_max)
        
        # Add transaction costs if available in the environment
        if hasattr(self.env, 'transaction_cost'):
            # Calculate transaction costs from position changes
            position_cols = [f'weight_{asset}' for asset in asset_names]
            results['position_change'] = results[position_cols].diff().abs().sum(axis=1)
            results['transaction_cost'] = results['position_change'] * self.env.transaction_cost
        
        # Store results dataframe
        self.results_df = results
        
        # Create visualizer and analytics
        benchmark_returns = self.env.benchmark_returns if hasattr(self.env, 'benchmark_returns') else None
        self.visualizer = BacktestVisualizer(self.results_df, benchmark_returns)
        self.analytics = PortfolioAnalytics(self.results_df, benchmark_returns)
    
    def get_results_dataframe(self):
        """
        Get backtest results as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with backtest results
        """
        if hasattr(self, 'results_df'):
            return self.results_df
        
        if not self.portfolio_values:
            logger.error("No backtest results to convert. Run backtest first.")
            return None
        
        # Create results dataframe if not already created
        self._create_results_dataframe()
        
        return self.results_df
    
    def generate_visualizations(self, output_dir=None, show=True):
        """
        Generate a simplified set of visualizations.
        
        Args:
            output_dir (str, optional): Directory to save visualizations
            show (bool): Whether to display the visualizations
            
        Returns:
            dict: Dictionary of matplotlib figures
        """
        if not hasattr(self, 'visualizer') or self.visualizer is None:
            if not self.portfolio_values:
                logger.error("No backtest results to visualize. Run backtest first.")
                return None
            self._create_results_dataframe()
        
        figures = {}
        
        # Generate only performance dashboard (cumulative returns, drawdowns, etc.)
        figures['performance'] = self.visualizer.create_performance_dashboard(
            show_benchmark=hasattr(self.env, 'benchmark_returns') and self.env.benchmark_returns is not None
        )
        
        # Generate risk analysis
        figures['risk'] = self.visualizer.create_risk_analysis()
        
        # Save figures if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for name, fig in figures.items():
                filepath = os.path.join(output_dir, f'{name}_{timestamp}.png')
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved {name} visualization to {filepath}")
        
        # Show figures if requested
        if show:
            for fig in figures.values():
                plt.figure(fig.number)
                plt.show()
        
        return figures
    
    def generate_portfolio_analytics(self, output_dir=None, show=True):
        """
        Generate portfolio manager-focused analytics across time horizons.
        
        Args:
            output_dir (str, optional): Directory to save visualizations
            show (bool): Whether to display the visualizations
            
        Returns:
            dict: Dictionary of matplotlib figures
        """
        if not hasattr(self, 'analytics') or self.analytics is None:
            if not self.portfolio_values:
                logger.error("No backtest results for analytics. Run backtest first.")
                return None
            self._create_results_dataframe()
        
        figures = {}
        
        # Only keep the most useful visualizations
        # Generate PnL time horizon dashboard
        figures['pnl_horizons'] = self.analytics.create_pnl_time_horizon_dashboard(
            show_plot=show
        )
        
        # Generate return distribution dashboard
        figures['return_distributions'] = self.analytics.create_return_distribution_dashboard(
            show_plot=show
        )
        
        # Save figures if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for name, fig in figures.items():
                filepath = os.path.join(output_dir, f'{name}_{timestamp}.png')
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved {name} analytics to {filepath}")
            
            # Generate and save metrics CSV only (no HTML to avoid formatting issues)
            metrics_table = self.analytics.create_performance_metrics_table()
            csv_path = os.path.join(output_dir, f'metrics_{timestamp}.csv')
            metrics_table.to_csv(csv_path, index=False)
            logger.info(f"Performance metrics saved to {csv_path}")
        
        return figures
    
    def create_performance_report(self, output_dir, include_visualizations=True, include_analytics=True):
        """
        Create a comprehensive performance report with metrics, visualizations, and analytics.
        
        Args:
            output_dir (str): Directory to save report
            include_visualizations (bool): Whether to include visualizations
            include_analytics (bool): Whether to include portfolio analytics
            
        Returns:
            str: Path to the generated report
        """
        if not self.portfolio_values:
            logger.error("No backtest results to report. Run backtest first.")
            return None
        
        # Ensure we have results dataframe, visualizer and analytics
        if not hasattr(self, 'results_df') or not hasattr(self, 'visualizer') or not hasattr(self, 'analytics'):
            self._create_results_dataframe()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate all metrics
        metrics = calculate_performance_metrics(
            returns=self.returns,
            portfolio_values=self.portfolio_values,
            benchmark_returns=self.env.benchmark_returns if hasattr(self.env, 'benchmark_returns') else None
        )
        
        # Create report filename
        report_path = os.path.join(output_dir, f'performance_report_{timestamp}.md')
        
        # Generate report content
        with open(report_path, 'w') as f:
            # Write header
            f.write(f"# Trading Strategy Performance Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            f.write("## Performance Summary\n\n")
            f.write(f"- **Total Return:** {metrics['total_return']:.2%}\n")
            f.write(f"- **Annualized Return:** {metrics['annualized_return']:.2%}\n")
            f.write(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"- **Maximum Drawdown:** {metrics['max_drawdown']:.2%}\n")
            f.write(f"- **Volatility:** {metrics['volatility']:.2%}\n")
            f.write(f"- **Calmar Ratio:** {metrics['calmar_ratio']:.2f}\n")
            f.write(f"- **Sortino Ratio:** {metrics['sortino_ratio']:.2f}\n\n")
            
            # Add performance across time periods
            f.write("## Performance Across Time Horizons\n\n")
            metrics_table = self.analytics.create_performance_metrics_table()
            
            # Convert DataFrame to markdown table
            f.write(metrics_table.to_markdown(index=False) + "\n\n")
            
            # Write detailed metrics
            f.write("## Detailed Metrics\n\n")
            f.write("### Return Metrics\n\n")
            f.write(f"- **Total Return:** {metrics['total_return']:.2%}\n")
            f.write(f"- **Annualized Return:** {metrics['annualized_return']:.2%}\n")
            f.write(f"- **Volatility:** {metrics['volatility']:.2%}\n")
            f.write(f"- **Skewness:** {metrics['skewness']:.2f}\n")
            f.write(f"- **Kurtosis:** {metrics['kurtosis']:.2f}\n\n")
            
            f.write("### Risk Metrics\n\n")
            f.write(f"- **Maximum Drawdown:** {metrics['max_drawdown']:.2%}\n")
            f.write(f"- **VaR (95%):** {metrics['var_95']:.2%}\n")
            f.write(f"- **CVaR (95%):** {metrics['cvar_95']:.2%}\n")
            f.write(f"- **Win Rate:** {metrics['win_rate']:.2%}\n")
            f.write(f"- **Gain/Loss Ratio:** {metrics['gain_loss_ratio']:.2f}\n\n")
            
            f.write("### Risk-Adjusted Metrics\n\n")
            f.write(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"- **Sortino Ratio:** {metrics['sortino_ratio']:.2f}\n")
            f.write(f"- **Calmar Ratio:** {metrics['calmar_ratio']:.2f}\n")
            f.write(f"- **Omega Ratio:** {metrics['omega_ratio']:.2f}\n\n")
            
            # Add benchmark comparison if available
            if hasattr(self.env, 'benchmark_returns') and self.env.benchmark_returns is not None:
                f.write("### Benchmark Comparison\n\n")
                if 'information_ratio' in metrics:
                    f.write(f"- **Information Ratio:** {metrics['information_ratio']:.2f}\n")
                if 'beta' in metrics:
                    f.write(f"- **Beta:** {metrics['beta']:.2f}\n")
                if 'alpha' in metrics:
                    f.write(f"- **Alpha:** {metrics['alpha']:.2%}\n")
                if 'r_squared' in metrics:
                    f.write(f"- **R-Squared:** {metrics['r_squared']:.2f}\n\n")
            
            # Add visualization references if included
            if include_visualizations:
                # Generate standard visualizations
                figures = self.generate_visualizations(output_dir=output_dir, show=False)
                
                f.write("## Visualizations\n\n")
                
                # For each visualization type, create a link to the image
                viz_types = {
                    'performance': 'Performance Dashboard',
                    'risk': 'Risk Analysis'
                }
                
                for viz_type, viz_name in viz_types.items():
                    viz_path = os.path.join(output_dir, f'{viz_type}_{timestamp}.png')
                    # Add a link to the image in the report
                    f.write(f"### {viz_name}\n\n")
                    f.write(f"![{viz_name}]({os.path.basename(viz_path)})\n\n")
            
            # Add portfolio analytics if included
            if include_analytics:
                # Generate portfolio analytics
                analytics_figures = self.generate_portfolio_analytics(output_dir=output_dir, show=False)
                
                f.write("## Portfolio Manager Analytics\n\n")
                
                # For each analytics type, create a link to the image
                analytics_types = {
                    'pnl_horizons': 'PnL Analysis by Time Horizon',
                    'return_distributions': 'Return Distribution Analysis'
                }
                
                for analytics_type, analytics_name in analytics_types.items():
                    analytics_path = os.path.join(output_dir, f'{analytics_type}_{timestamp}.png')
                    # Add a link to the image in the report
                    f.write(f"### {analytics_name}\n\n")
                    f.write(f"![{analytics_name}]({os.path.basename(analytics_path)})\n\n")
        
        logger.info(f"Performance report created at {report_path}")
        return report_path

    def plot_results(self, benchmark_returns=None, figsize=(12, 8)):
        """
        Plot backtest results (legacy method for compatibility).
        
        Args:
            benchmark_returns (list): List of benchmark returns for comparison
            figsize (tuple): Figure size
            
        Returns:
            tuple: Matplotlib figures
        """
        logger.info("Using legacy plot_results method. Consider using generate_visualizations() for enhanced visuals.")
        
        if not self.portfolio_values:
            logger.error("No backtest results to plot. Run backtest first.")
            return None
        
        # Convert portfolio values to cumulative returns
        cumulative_returns = np.array(self.portfolio_values) / self.portfolio_values[0] - 1
        
        # Create figure for portfolio performance
        fig1, axes1 = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot cumulative returns
        axes1[0].plot(cumulative_returns, label='Strategy')
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns)) - 1
            axes1[0].plot(benchmark_cumulative, label='Benchmark')
        
        axes1[0].set_title('Cumulative Returns')
        axes1[0].set_ylabel('Return (%)')
        axes1[0].legend()
        axes1[0].grid(True)
        
        # Plot drawdowns
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (1 + running_max)
        axes1[1].fill_between(range(len(drawdowns)), 0, drawdowns, color='r', alpha=0.3)
        axes1[1].set_title('Drawdowns')
        axes1[1].set_ylabel('Drawdown (%)')
        axes1[1].set_xlabel('Trading Days')
        axes1[1].grid(True)
        
        fig1.tight_layout()
        
        # Create figure for position allocation
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        # Convert positions to array for plotting
        positions_array = np.array(self.positions)
        
        # Get asset names from environment
        if hasattr(self.env, 'price_data') and hasattr(self.env.price_data, 'columns'):
            asset_names = self.env.price_data.columns
        else:
            asset_names = [f'Asset {i+1}' for i in range(positions_array.shape[1])]
        
        # Plot positions over time
        for i, asset in enumerate(asset_names):
            ax2.plot(positions_array[:, i], label=asset)
        
        ax2.set_title('Portfolio Allocation Over Time')
        ax2.set_ylabel('Weight')
        ax2.set_xlabel('Trading Days')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True)
        
        fig2.tight_layout()
        
        return fig1, fig2