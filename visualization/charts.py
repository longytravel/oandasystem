"""
Interactive trade visualization charts using Plotly.

Creates candlestick charts with trade entry/exit markers,
SL/TP lines, and equity curve subplots.
"""
from typing import List, Optional
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.base import Trade, SignalType
from backtesting.engine import BacktestResult


class TradeChart:
    """
    Interactive chart generator for backtest visualization.

    Features:
    - Candlestick OHLC chart
    - Trade entry markers (green up arrow = BUY, red down arrow = SELL)
    - Trade exit markers (star = TP, X = SL)
    - SL/TP horizontal lines during each trade
    - Equity curve subplot
    - Interactive zoom/pan/hover
    - Save as HTML or PNG
    """

    def __init__(
        self,
        title: str = "Trade Chart",
        height: int = 900,
        show_equity: bool = True,
        hide_weekends: bool = True,
        show_profit: bool = True,
    ):
        """
        Initialize chart generator.

        Args:
            title: Chart title
            height: Chart height in pixels
            show_equity: Whether to show equity/profit curve subplot
            hide_weekends: Whether to hide weekend gaps
            show_profit: If True, show profit (starting from 0) instead of equity
        """
        self.title = title
        self.height = height
        self.show_equity = show_equity
        self.hide_weekends = hide_weekends
        self.show_profit = show_profit

    def create_chart(
        self,
        df: pd.DataFrame,
        result: BacktestResult,
        show_sl_tp_lines: bool = True,
    ) -> go.Figure:
        """
        Create interactive chart from backtest results.

        Args:
            df: OHLCV DataFrame with datetime index
            result: BacktestResult from backtesting engine
            show_sl_tp_lines: Whether to show SL/TP horizontal lines

        Returns:
            Plotly Figure object
        """
        trades = result.trades
        equity_curve = result.equity_curve

        # Create subplots
        if self.show_equity:
            bottom_title = "Profit Curve" if self.show_profit else "Equity Curve"
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=("Price Chart", bottom_title)
            )
        else:
            fig = make_subplots(rows=1, cols=1)

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            ),
            row=1, col=1
        )

        # Collect trade markers
        buy_entries = []
        sell_entries = []
        tp_exits = []
        sl_exits = []
        other_exits = []

        for trade in trades:
            entry_data = {
                'time': trade.entry_time,
                'price': trade.entry_price,
                'text': f"Entry: {trade.entry_price:.5f}<br>SL: {trade.stop_loss:.5f}<br>TP: {trade.take_profit:.5f}"
            }

            exit_data = {
                'time': trade.exit_time,
                'price': trade.exit_price,
                'text': f"Exit: {trade.exit_price:.5f}<br>PnL: {trade.pnl:+.2f}<br>Reason: {trade.exit_reason}"
            }

            if trade.direction == SignalType.BUY:
                buy_entries.append(entry_data)
            else:
                sell_entries.append(entry_data)

            if trade.exit_reason == "TP":
                tp_exits.append(exit_data)
            elif trade.exit_reason in ("SL", "BE", "TRAIL"):
                sl_exits.append(exit_data)
            else:
                other_exits.append(exit_data)

        # Add BUY entry markers (green up arrows)
        if buy_entries:
            fig.add_trace(
                go.Scatter(
                    x=[e['time'] for e in buy_entries],
                    y=[e['price'] for e in buy_entries],
                    mode='markers',
                    name='BUY Entry',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00c853',
                        line=dict(width=1, color='white')
                    ),
                    text=[e['text'] for e in buy_entries],
                    hovertemplate='%{text}<extra>BUY</extra>'
                ),
                row=1, col=1
            )

        # Add SELL entry markers (red down arrows)
        if sell_entries:
            fig.add_trace(
                go.Scatter(
                    x=[e['time'] for e in sell_entries],
                    y=[e['price'] for e in sell_entries],
                    mode='markers',
                    name='SELL Entry',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='#ff1744',
                        line=dict(width=1, color='white')
                    ),
                    text=[e['text'] for e in sell_entries],
                    hovertemplate='%{text}<extra>SELL</extra>'
                ),
                row=1, col=1
            )

        # Add TP exit markers (star)
        if tp_exits:
            fig.add_trace(
                go.Scatter(
                    x=[e['time'] for e in tp_exits],
                    y=[e['price'] for e in tp_exits],
                    mode='markers',
                    name='Take Profit',
                    marker=dict(
                        symbol='star',
                        size=14,
                        color='#00e676',
                        line=dict(width=1, color='white')
                    ),
                    text=[e['text'] for e in tp_exits],
                    hovertemplate='%{text}<extra>TP</extra>'
                ),
                row=1, col=1
            )

        # Add SL exit markers (X)
        if sl_exits:
            fig.add_trace(
                go.Scatter(
                    x=[e['time'] for e in sl_exits],
                    y=[e['price'] for e in sl_exits],
                    mode='markers',
                    name='Stop Loss',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color='#ff5252',
                        line=dict(width=2)
                    ),
                    text=[e['text'] for e in sl_exits],
                    hovertemplate='%{text}<extra>SL</extra>'
                ),
                row=1, col=1
            )

        # Add other exit markers (circle)
        if other_exits:
            fig.add_trace(
                go.Scatter(
                    x=[e['time'] for e in other_exits],
                    y=[e['price'] for e in other_exits],
                    mode='markers',
                    name='Exit (Other)',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='#ffab00',
                        line=dict(width=1, color='white')
                    ),
                    text=[e['text'] for e in other_exits],
                    hovertemplate='%{text}<extra>EXIT</extra>'
                ),
                row=1, col=1
            )

        # Add SL/TP lines for each trade
        if show_sl_tp_lines:
            for trade in trades:
                # SL line (red dashed)
                fig.add_shape(
                    type="line",
                    x0=trade.entry_time,
                    x1=trade.exit_time,
                    y0=trade.stop_loss,
                    y1=trade.stop_loss,
                    line=dict(color="#ff5252", width=1, dash="dash"),
                    row=1, col=1
                )

                # TP line (green dashed)
                fig.add_shape(
                    type="line",
                    x0=trade.entry_time,
                    x1=trade.exit_time,
                    y0=trade.take_profit,
                    y1=trade.take_profit,
                    line=dict(color="#00e676", width=1, dash="dash"),
                    row=1, col=1
                )

        # Add equity/profit curve
        if self.show_equity and equity_curve is not None and len(equity_curve) > 0:
            # Create time index for equity curve based on trade exits
            equity_times = [df.index[0]]  # Start
            for trade in trades:
                equity_times.append(trade.exit_time)

            # Ensure same length
            if len(equity_times) > len(equity_curve):
                equity_times = equity_times[:len(equity_curve)]
            elif len(equity_times) < len(equity_curve):
                # Pad with last trade time
                while len(equity_times) < len(equity_curve):
                    equity_times.append(equity_times[-1])

            # Convert to profit if requested (start from 0)
            starting_capital = equity_curve.iloc[0]
            if self.show_profit:
                plot_values = equity_curve.values - starting_capital
                curve_name = 'Profit'
                zero_line = 0
            else:
                plot_values = equity_curve.values
                curve_name = 'Equity'
                zero_line = starting_capital

            # Determine color based on profit
            final_pnl = equity_curve.iloc[-1] - starting_capital
            equity_color = '#00c853' if final_pnl >= 0 else '#ff1744'

            fig.add_trace(
                go.Scatter(
                    x=equity_times,
                    y=plot_values,
                    mode='lines',
                    name=curve_name,
                    line=dict(color=equity_color, width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba({",".join(str(int(equity_color[i:i+2], 16)) for i in (1, 3, 5))}, 0.1)'
                ),
                row=2, col=1
            )

            # Add zero/starting line
            fig.add_hline(
                y=zero_line,
                line_dash="dash",
                line_color="gray",
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=self.title,
                x=0.5,
                font=dict(size=20)
            ),
            height=self.height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark',
            dragmode='zoom',  # Enable box zoom by default
        )

        # Hide weekends (Saturday and Sunday)
        if self.hide_weekends:
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # Hide weekends
                ],
                row=1, col=1
            )
            if self.show_equity:
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),
                    ],
                    row=2, col=1
                )

        # Update axes
        fig.update_xaxes(title_text="Date", row=2 if self.show_equity else 1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if self.show_equity:
            ylabel = "Profit" if self.show_profit else "Equity"
            fig.update_yaxes(title_text=ylabel, row=2, col=1)

        return fig

    def save_html(
        self,
        fig: go.Figure,
        filepath: str,
        auto_open: bool = False
    ) -> Path:
        """
        Save chart as interactive HTML file.

        Args:
            fig: Plotly Figure object
            filepath: Output file path
            auto_open: Open in browser after saving

        Returns:
            Path to saved file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), auto_open=auto_open)
        return path

    def save_png(
        self,
        fig: go.Figure,
        filepath: str,
        width: int = 1920,
        height: Optional[int] = None
    ) -> Path:
        """
        Save chart as PNG image.

        Note: Requires kaleido package: pip install kaleido

        Args:
            fig: Plotly Figure object
            filepath: Output file path
            width: Image width in pixels
            height: Image height (defaults to chart height)

        Returns:
            Path to saved file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(
            str(path),
            width=width,
            height=height or self.height
        )
        return path

    def show(self, fig: go.Figure):
        """Display chart in browser."""
        fig.show()


def create_quick_chart(
    df: pd.DataFrame,
    result: BacktestResult,
    title: str = "Backtest Results",
    save_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Quick helper to create and optionally save/show a trade chart.

    Args:
        df: OHLCV DataFrame
        result: BacktestResult from backtesting engine
        title: Chart title
        save_path: Optional path to save HTML file
        show: Whether to display the chart

    Returns:
        Plotly Figure object
    """
    chart = TradeChart(title=title)
    fig = chart.create_chart(df, result)

    if save_path:
        chart.save_html(fig, save_path)

    if show:
        chart.show(fig)

    return fig
