"""
Visualization module for GEM strategy analysis.

This module provides interactive visualization capabilities using Plotly
for analyzing strategy performance, allocations, and risk metrics.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger

from gem.config import GemConfig


class Visualizer:
    """
    Interactive visualization toolkit for investment strategy analysis.
    
    Provides methods for creating:
    - Performance comparison charts
    - Allocation timeline charts  
    - Risk analysis plots
    - Drawdown visualization
    - Correlation matrices
    - Rolling metrics charts
    """
    
    def __init__(self, config: GemConfig):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.default_template = "plotly_white"
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
    def plot_cumulative_returns(
        self,
        returns_data: Dict[str, pd.Series],
        title: str = "Cumulative Returns Comparison",
        show_drawdown: bool = True
    ) -> go.Figure:
        """
        Plot cumulative returns for multiple strategies.
        
        Args:
            returns_data: Dictionary of return series
            title: Chart title
            show_drawdown: Whether to show drawdown subplot
            
        Returns:
            Plotly figure
        """
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Cumulative Returns', 'Drawdown'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
            
        colors = iter(self.color_palette)
        
        for name, returns in returns_data.items():
            if returns.empty:
                continue
                
            # Calculate cumulative returns
            cumulative_returns = (1 + returns.fillna(0)).cumprod() - 1
            
            color = next(colors)
            
            # Add cumulative returns line
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns * 100,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=1 if show_drawdown else None,
                col=1 if show_drawdown else None
            )
            
            # Add drawdown if requested
            if show_drawdown:
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / (1 + running_max) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        mode='lines',
                        name=f'{name} Drawdown',
                        line=dict(color=color, width=1),
                        fill='tonexty' if name == list(returns_data.keys())[0] else None,
                        fillcolor=f'rgba{color[3:-1]}, 0.3)',
                        showlegend=False,
                        hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                    ),
                    row=2,
                    col=1
                )
                
        # Update layout
        fig.update_layout(
            title=title,
            template=self.default_template,
            height=600 if show_drawdown else 400,
            hovermode='x unified'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1 if show_drawdown else None)
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown (%)", row=2)
            
        return fig
        
    def plot_allocation_timeline(
        self,
        allocation_df: pd.DataFrame,
        title: str = "Asset Allocation Over Time"
    ) -> go.Figure:
        """
        Plot asset allocation timeline.
        
        Args:
            allocation_df: DataFrame with allocation data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Find allocation columns
        allocation_cols = [col for col in allocation_df.columns if col.startswith('allocation_')]
        
        if not allocation_cols:
            logger.warning("No allocation columns found")
            return go.Figure()
            
        fig = go.Figure()
        
        # Create stacked area chart
        cumulative = pd.Series(0, index=allocation_df.index)
        colors = iter(self.color_palette)
        
        for col in allocation_cols:
            asset_name = col.replace('allocation_', '').replace('_', ' ').title()
            allocation_pct = allocation_df[col] * 100
            
            color = next(colors)
            
            fig.add_trace(
                go.Scatter(
                    x=allocation_df.index,
                    y=cumulative + allocation_pct,
                    fill='tonexty',
                    mode='none',
                    name=asset_name,
                    fillcolor=color,
                    hovertemplate=f'{asset_name}: %{{customdata:.1f}}%<extra></extra>',
                    customdata=allocation_pct
                )
            )
            
            cumulative += allocation_pct
            
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Allocation (%)",
            template=self.default_template,
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
        
    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        metrics: List[str] = None,
        window: int = 252,
        title: str = "Rolling Performance Metrics"
    ) -> go.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Return series
            metrics: List of metrics to plot
            window: Rolling window size
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ['return', 'volatility', 'sharpe']
            
        from gem.analysis.metrics import PerformanceMetrics
        perf_metrics = PerformanceMetrics()
        
        # Calculate rolling metrics
        rolling_df = perf_metrics.calculate_rolling_metrics(returns, window, metrics)
        
        if rolling_df.empty:
            return go.Figure()
            
        # Create subplots
        n_metrics = len([col for col in rolling_df.columns if not col.isna().all()])
        fig = make_subplots(
            rows=n_metrics, cols=1,
            shared_xaxes=True,
            subplot_titles=[col.replace('rolling_', '').replace('_', ' ').title() 
                          for col in rolling_df.columns if not col.isna().all()],
            vertical_spacing=0.1
        )
        
        colors = iter(self.color_palette)
        row = 1
        
        for col in rolling_df.columns:
            if rolling_df[col].isna().all():
                continue
                
            color = next(colors)
            metric_name = col.replace('rolling_', '').replace('_', ' ').title()
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df[col],
                    mode='lines',
                    name=metric_name,
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            row += 1
            
        fig.update_layout(
            title=title,
            template=self.default_template,
            height=200 * n_metrics,
            hovermode='x unified'
        )
        
        return fig
        
    def plot_risk_return_scatter(
        self,
        strategy_metrics: Dict[str, Dict[str, float]],
        title: str = "Risk-Return Analysis"
    ) -> go.Figure:
        """
        Create risk-return scatter plot.
        
        Args:
            strategy_metrics: Dictionary of strategy metrics
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        x_values = []
        y_values = []
        names = []
        colors = []
        
        color_iter = iter(self.color_palette)
        
        for name, metrics in strategy_metrics.items():
            if 'volatility' in metrics and 'annualized_return' in metrics:
                x_values.append(metrics['volatility'] * 100)
                y_values.append(metrics['annualized_return'] * 100)
                names.append(name)
                colors.append(next(color_iter))
                
        if not x_values:
            return fig
            
        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers+text',
                text=names,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='%{text}<br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Volatility (%)",
            yaxis_title="Annualized Return (%)",
            template=self.default_template,
            showlegend=False
        )
        
        return fig
        
    def plot_monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap"
    ) -> go.Figure:
        """
        Create monthly returns heatmap.
        
        Args:
            returns: Return series
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Convert to monthly returns if not already
        if len(returns) > 100:  # Assume daily data if > 100 observations
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        else:
            monthly_returns = returns
            
        # Create pivot table
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot_data = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).first().unstack()
        
        # Convert to percentage and format
        pivot_data = pivot_data * 100
        
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=[f'{month:02d}' for month in pivot_data.columns],
                y=pivot_data.index,
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>',
                colorbar=dict(title="Return (%)")
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Year",
            template=self.default_template
        )
        
        return fig
        
    def plot_correlation_matrix(
        self,
        returns_data: Dict[str, pd.Series],
        title: str = "Return Correlations"
    ) -> go.Figure:
        """
        Plot correlation matrix of returns.
        
        Args:
            returns_data: Dictionary of return series
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Create DataFrame and calculate correlations
        df = pd.DataFrame(returns_data)
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Correlation")
            )
        )
        
        fig.update_layout(
            title=title,
            template=self.default_template,
            width=600,
            height=600
        )
        
        return fig
        
    def create_performance_dashboard(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        allocation_df: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            allocation_df: Allocation DataFrame
            
        Returns:
            Plotly figure with multiple subplots
        """
        # Determine subplot configuration
        n_rows = 3 if allocation_df is not None else 2
        
        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=[
                'Cumulative Returns', 'Rolling Volatility',
                'Drawdown', 'Monthly Returns Distribution'
            ] + (['Asset Allocation', 'Risk Metrics'] if allocation_df is not None else []),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ] + ([[{"secondary_y": False}, {"secondary_y": False}]] if allocation_df is not None else []),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Cumulative Returns
        returns_data = {'Strategy': strategy_returns}
        if benchmark_returns is not None:
            returns_data['Benchmark'] = benchmark_returns
            
        colors = iter(self.color_palette)
        
        for name, returns in returns_data.items():
            cumulative = (1 + returns.fillna(0)).cumprod() - 1
            color = next(colors)
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative * 100,
                    name=name,
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                row=1, col=1
            )
            
        # 2. Rolling Volatility
        rolling_vol = strategy_returns.rolling(252).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name='Rolling Volatility',
                line=dict(color=self.color_palette[0]),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Drawdown
        cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
        running_max = cumulative_strategy.expanding().max()
        drawdown = (cumulative_strategy - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                name='Drawdown',
                fill='tonexty',
                line=dict(color=self.color_palette[1]),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Monthly Returns Distribution
        monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        fig.add_trace(
            go.Histogram(
                x=monthly_returns,
                name='Monthly Returns',
                nbinsx=20,
                marker_color=self.color_palette[2],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Asset Allocation (if provided)
        if allocation_df is not None and n_rows > 2:
            allocation_cols = [col for col in allocation_df.columns if col.startswith('allocation_')]
            if allocation_cols:
                cumulative_alloc = pd.Series(0, index=allocation_df.index)
                
                for i, col in enumerate(allocation_cols):
                    asset_name = col.replace('allocation_', '').replace('_', ' ').title()
                    allocation_pct = allocation_df[col] * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=allocation_df.index,
                            y=cumulative_alloc + allocation_pct,
                            fill='tonexty',
                            name=asset_name,
                            fillcolor=self.color_palette[i % len(self.color_palette)],
                            line=dict(width=0),
                            showlegend=False
                        ),
                        row=3, col=1
                    )
                    
                    cumulative_alloc += allocation_pct
                    
        # Update layout
        fig.update_layout(
            title="Strategy Performance Dashboard",
            template=self.default_template,
            height=800 if n_rows > 2 else 600,
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        if n_rows > 2:
            fig.update_yaxes(title_text="Allocation (%)", row=3, col=1)
            
        return fig
        
    def save_figure(
        self,
        fig: go.Figure,
        filename: str,
        format: str = "html",
        width: int = 1200,
        height: int = 800
    ) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format ('html', 'png', 'pdf', 'svg')
            width: Image width (for image formats)
            height: Image height (for image formats)
        """
        output_path = self.config.output_directory / f"{filename}.{format}"
        
        try:
            if format == "html":
                fig.write_html(str(output_path))
            elif format in ["png", "pdf", "svg"]:
                fig.write_image(str(output_path), width=width, height=height)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Figure saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")
            raise