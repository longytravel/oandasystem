"""
Example: Full 30+ Parameter EA Configuration for Staged Optimization.

This shows how you'd configure a complex EA with many parameters
organized into logical groups for staged optimization.

The staged optimizer will:
1. Optimize each group sequentially (lock winners)
2. Final pass with tight ranges around all locked values
"""

# =============================================================================
# EXAMPLE: Full RSI Divergence EA with 35 Parameters
# =============================================================================

FULL_EA_PARAMETER_GROUPS = {

    # =========================================================================
    # GROUP 1: SIGNAL GENERATION (8 params)
    # Core indicator and pattern detection settings
    # =========================================================================
    'signal': {
        'rsi_period': {
            'values': [5, 7, 9, 11, 14, 18, 21, 25, 30],
            'default': 14,
            'description': 'RSI calculation period'
        },
        'rsi_overbought': {
            'values': [65, 70, 75, 80, 85],
            'default': 70,
            'description': 'RSI overbought level'
        },
        'rsi_oversold': {
            'values': [15, 20, 25, 30, 35],
            'default': 30,
            'description': 'RSI oversold level'
        },
        'swing_strength': {
            'values': [3, 5, 7, 9, 11, 13],
            'default': 5,
            'description': 'Bars either side for swing detection'
        },
        'min_rsi_diff': {
            'values': [2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0],
            'default': 5.0,
            'description': 'Minimum RSI difference for divergence'
        },
        'min_bars_between': {
            'values': [3, 5, 8, 12, 16, 20, 25],
            'default': 8,
            'description': 'Minimum bars between swings'
        },
        'max_bars_between': {
            'values': [40, 60, 80, 100, 120, 150],
            'default': 80,
            'description': 'Maximum bars between swings'
        },
        'require_confirmation': {
            'values': [True, False],
            'default': False,
            'description': 'Require candle confirmation'
        },
    },

    # =========================================================================
    # GROUP 2: TREND FILTERS (6 params)
    # Higher timeframe and trend alignment filters
    # =========================================================================
    'trend': {
        'use_trend_filter': {
            'values': [True, False],
            'default': True,
            'description': 'Enable trend filter'
        },
        'trend_ma_period': {
            'values': [50, 100, 150, 200],
            'default': 200,
            'description': 'MA period for trend detection'
        },
        'trend_ma_type': {
            'values': ['SMA', 'EMA'],
            'default': 'EMA',
            'description': 'Moving average type'
        },
        'trade_with_trend_only': {
            'values': [True, False],
            'default': False,
            'description': 'Only trade in trend direction'
        },
        'min_trend_strength': {
            'values': [0.0, 0.5, 1.0, 1.5, 2.0],
            'default': 0.0,
            'description': 'Minimum ATR distance from MA'
        },
        'htf_confirmation': {
            'values': [True, False],
            'default': False,
            'description': 'Require higher TF confirmation'
        },
    },

    # =========================================================================
    # GROUP 3: ENTRY FILTERS (7 params)
    # Time, volatility, and market condition filters
    # =========================================================================
    'filters': {
        'trade_start_hour': {
            'values': [0, 2, 4, 6, 8, 10],
            'default': 6,
            'description': 'Start trading hour (broker time)'
        },
        'trade_end_hour': {
            'values': [16, 18, 20, 22, 24],
            'default': 20,
            'description': 'Stop trading hour'
        },
        'trade_monday': {
            'values': [True, False],
            'default': True,
            'description': 'Trade on Monday'
        },
        'trade_friday': {
            'values': [True, False],
            'default': True,
            'description': 'Trade on Friday'
        },
        'min_price_slope': {
            'values': [0.0, 5.0, 10.0, 15.0, 25.0, 35.0],
            'default': 5.0,
            'description': 'Minimum price slope angle'
        },
        'max_price_slope': {
            'values': [40.0, 50.0, 65.0, 80.0, 90.0],
            'default': 65.0,
            'description': 'Maximum price slope angle'
        },
        'max_spread_pips': {
            'values': [1.5, 2.0, 2.5, 3.0, 5.0],
            'default': 2.5,
            'description': 'Maximum allowed spread'
        },
    },

    # =========================================================================
    # GROUP 4: RISK MANAGEMENT (6 params)
    # Position sizing, stop loss, take profit
    # =========================================================================
    'risk': {
        'risk_percent': {
            'values': [0.5, 1.0, 1.5, 2.0, 3.0],
            'default': 1.0,
            'description': 'Risk per trade as % of equity'
        },
        'stop_loss_pips': {
            'values': [15, 20, 25, 30, 40, 50, 75, 100],
            'default': 30,
            'description': 'Stop loss in pips'
        },
        'stop_loss_atr_mult': {
            'values': [0.0, 1.0, 1.5, 2.0, 2.5],
            'default': 0.0,
            'description': 'SL as ATR multiple (0=use pips)'
        },
        'tp_multiplier': {
            'values': [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
            'default': 1.5,
            'description': 'TP as multiple of SL'
        },
        'max_daily_trades': {
            'values': [1, 2, 3, 5, 10, 99],
            'default': 5,
            'description': 'Maximum trades per day'
        },
        'max_daily_loss_pct': {
            'values': [1.0, 2.0, 3.0, 5.0, 10.0],
            'default': 3.0,
            'description': 'Daily loss limit as %'
        },
    },

    # =========================================================================
    # GROUP 5: TRADE MANAGEMENT (8 params)
    # Trailing stops, breakeven, partial closes
    # =========================================================================
    'management': {
        'use_trailing_stop': {
            'values': [True, False],
            'default': False,
            'description': 'Enable trailing stop'
        },
        'trail_start_pips': {
            'values': [10, 15, 20, 30, 40, 50],
            'default': 20,
            'description': 'Profit to start trailing'
        },
        'trail_step_pips': {
            'values': [5, 10, 15, 20],
            'default': 10,
            'description': 'Trailing step size'
        },
        'use_breakeven': {
            'values': [True, False],
            'default': True,
            'description': 'Move SL to breakeven'
        },
        'breakeven_trigger_pips': {
            'values': [10, 15, 20, 25, 30, 40],
            'default': 20,
            'description': 'Profit to trigger breakeven'
        },
        'breakeven_offset_pips': {
            'values': [0, 1, 2, 3, 5],
            'default': 1,
            'description': 'Pips above entry for BE'
        },
        'use_partial_close': {
            'values': [True, False],
            'default': False,
            'description': 'Enable partial position close'
        },
        'partial_close_pct': {
            'values': [25, 33, 50, 66, 75],
            'default': 50,
            'description': 'Percent to close at target'
        },
    },
}


# =============================================================================
# Calculate total parameter space
# =============================================================================

def calculate_space_size(config: dict) -> dict:
    """Calculate parameter space statistics."""
    stats = {
        'groups': {},
        'total_params': 0,
        'total_combinations': 1,
    }

    for group_name, params in config.items():
        group_combos = 1
        n_params = len(params)

        for param_name, param_config in params.items():
            n_values = len(param_config['values'])
            group_combos *= n_values

        stats['groups'][group_name] = {
            'n_params': n_params,
            'combinations': group_combos,
        }
        stats['total_params'] += n_params
        stats['total_combinations'] *= group_combos

    return stats


if __name__ == '__main__':
    stats = calculate_space_size(FULL_EA_PARAMETER_GROUPS)

    print("\n" + "="*70)
    print("FULL EA PARAMETER SPACE ANALYSIS")
    print("="*70)

    for group_name, group_stats in stats['groups'].items():
        print(f"{group_name:<20} {group_stats['n_params']:>3} params -> {group_stats['combinations']:>15,} combinations")

    print("-"*70)
    print(f"{'TOTAL':<20} {stats['total_params']:>3} params -> {stats['total_combinations']:>15,} combinations")
    print("="*70)

    # Staged approach analysis
    print("\nSTAGED OPTIMIZATION APPROACH:")
    print("-"*70)

    trials_per_stage = 5000
    total_stage_trials = trials_per_stage * len(stats['groups'])
    final_trials = 10000
    total_trials = total_stage_trials + final_trials

    print(f"Stages: {len(stats['groups'])} × {trials_per_stage:,} trials = {total_stage_trials:,}")
    print(f"Final:  {final_trials:,} trials (tight ranges)")
    print(f"Total:  {total_trials:,} trials")
    print()
    print(f"Exhaustive search would need: {stats['total_combinations']:,} trials")
    print(f"Staged approach uses:         {total_trials:,} trials")
    print(f"Reduction factor:             {stats['total_combinations']/total_trials:,.0f}x")
    print("="*70)
