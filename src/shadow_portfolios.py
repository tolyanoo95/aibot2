"""
Shadow analysis: check what alternative configs would do at each bot decision point.
Uses the SAME bot logic — only varies parameters (threshold, SL/TP, trailing, etc).
No separate state management — piggybacks on real bot decisions.
"""

import json
import logging
import os
from collections import defaultdict
from itertools import product

logger = logging.getLogger(__name__)

COMMISSION_PCT = 0.08

GRID_PARAMS = {
    'sl_atr': [1.0, 1.5, 2.0],
    'tp_atr': [1.5, 2.0, 2.5, 3.0],
    'trail_activation': [1.5, 2.0, 2.5, 0],  # 0 = disabled
    'timeout': [12, 18, 24],
}


class ShadowAnalyzer:
    """Analyzes what alternative configs would do at each real bot decision."""

    def __init__(self, config_path: str = "shadow_configs.json", base_config=None):
        self.configs = []
        self._base = base_config
        self._grid_results = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0,
                                                    'long_n': 0, 'long_pnl': 0.0,
                                                    'short_n': 0, 'short_pnl': 0.0})
        self._tick = 0
        self._summary_interval = 12  # every ~1 hour

        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    self.configs = json.load(f)
                logger.info("Loaded %d shadow configs", len(self.configs))
            except Exception as exc:
                logger.error("Failed to load shadow configs: %s", exc)

    @property
    def active(self):
        return len(self.configs) > 0

    def check_open(self, symbol, direction, confidence, regime, atr, entry_price,
                   ml_disagreement, effective_threshold):
        """Called when bot makes open/skip decision. Returns log lines."""
        lines = []
        would_open = []
        would_skip = []

        eff_conf = confidence
        if ml_disagreement >= 0.5:
            eff_conf *= 0.60
        elif ml_disagreement > 0:
            eff_conf *= 0.85

        for cfg in self.configs:
            name = cfg.get('name', '?')

            if cfg.get('inverse', False):
                would_skip.append(f"{name}(inv)")
                continue

            skip_regimes = [tuple(x) for x in cfg.get('skip_regimes', [])]
            if (direction, regime) in skip_regimes:
                would_skip.append(f"{name}(regime)")
                continue

            thr = cfg.get('threshold', getattr(self._base, 'PREDICTION_THRESHOLD', 0.80))
            if eff_conf < thr:
                would_skip.append(f"{name}(conf<{thr:.0%})")
                continue

            sl_m = cfg.get('sl_atr', getattr(self._base, 'SL_ATR_MULTIPLIER', 1.5))
            tp_m = cfg.get('tp_atr', getattr(self._base, 'TP_ATR_MULTIPLIER', 3.0))

            regime_tp = cfg.get('regime_tp', {})
            regime_sl = cfg.get('regime_sl', {})
            if regime in regime_tp:
                tp_m = regime_tp[regime]
            if regime in regime_sl:
                sl_m = regime_sl[regime]

            sl_dist = atr * sl_m
            tp_dist = atr * tp_m
            min_sl = entry_price * 0.3 / 100
            if sl_dist < min_sl:
                sl_dist = min_sl

            if direction == 'LONG':
                alt_sl = entry_price - sl_dist
                alt_tp = entry_price + tp_dist
            else:
                alt_sl = entry_price + sl_dist
                alt_tp = entry_price - tp_dist

            would_open.append(f"{name}(sl={alt_sl:.6g},tp={alt_tp:.6g})")

        if would_open or would_skip:
            lines.append(
                f"SHADOW_OPEN {symbol} {direction} | conf={confidence:.1%} regime={regime} | "
                f"open: {','.join(would_open) if would_open else 'none'} | "
                f"skip: {','.join(would_skip) if would_skip else 'none'}"
            )

        return lines

    def check_close(self, symbol, direction, entry_price, atr, reason, pnl,
                    bars_held, trade_history):
        """
        Called when a real trade closes. Calculates what each alternative config
        would have done using the same price history.

        trade_history: list of dicts with keys 'high', 'low', 'close', 'atr'
        for each bar the trade was open (from CANDLE_1M or SCAN data).
        """
        lines = []
        alt_results = []

        for cfg in self.configs:
            name = cfg.get('name', '?')
            if cfg.get('inverse', False):
                continue

            alt = self._calc_alt_exit(
                cfg, direction, entry_price, atr, trade_history,
            )
            if alt:
                alt_results.append(f"{name}({alt['reason']} {alt['pnl']:+.2f}%,bar={alt['bar']})")

        # Grid search: all SL/TP/trail/timeout combos
        grid_alts = self._grid_check(direction, entry_price, atr, trade_history)

        if alt_results:
            lines.append(
                f"SHADOW_CLOSE {symbol} {direction} | reason={reason} pnl={pnl:+.2f}% bars={bars_held} | "
                f"alt: {' '.join(alt_results)}"
            )

        return lines

    def _calc_alt_exit(self, cfg, direction, entry_price, entry_atr, history):
        """Calculate exit for one alternative config given trade price history."""
        sl_m = cfg.get('sl_atr', getattr(self._base, 'SL_ATR_MULTIPLIER', 1.5))
        tp_m = cfg.get('tp_atr', getattr(self._base, 'TP_ATR_MULTIPLIER', 3.0))
        trail_act = cfg.get('trail_activation', getattr(self._base, 'TRAILING_ACTIVATION_ATR', 2.0))
        trail_dist = cfg.get('trail_distance', getattr(self._base, 'TRAILING_DISTANCE_ATR', 1.5))
        trail_on = cfg.get('trail_enabled', True)
        timeout = cfg.get('timeout', getattr(self._base, 'MAX_HOLD_BARS', 18))
        exit_mode = cfg.get('exit_mode', 'high_low')

        regime_tp = cfg.get('regime_tp', {})
        regime_sl = cfg.get('regime_sl', {})

        sl_dist = entry_atr * sl_m
        tp_dist = entry_atr * tp_m
        min_sl = entry_price * 0.3 / 100
        if sl_dist < min_sl:
            sl_dist = min_sl

        if direction == 'LONG':
            sl, tp = entry_price - sl_dist, entry_price + tp_dist
        else:
            sl, tp = entry_price + sl_dist, entry_price - tp_dist

        current_sl = sl
        best_price = entry_price
        trailing_active = False

        for bar_idx, bar in enumerate(history, 1):
            hi = bar.get('high', bar.get('close', entry_price))
            lo = bar.get('low', bar.get('close', entry_price))
            cl = bar.get('close', entry_price)
            bar_atr = bar.get('atr', entry_atr)

            if direction == 'LONG':
                best_price = max(best_price, hi)
            else:
                best_price = min(best_price, lo)

            # Trailing
            if trail_on and bar_atr > 0:
                act = bar_atr * trail_act
                dist = bar_atr * trail_dist
                if direction == 'LONG' and best_price >= entry_price + act:
                    trailing_active = True
                    new_sl = best_price - dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                elif direction == 'SHORT' and best_price <= entry_price - act:
                    trailing_active = True
                    new_sl = best_price + dist
                    if new_sl < current_sl:
                        current_sl = new_sl

            check_hi = hi if exit_mode == 'high_low' else cl
            check_lo = lo if exit_mode == 'high_low' else cl

            # TP
            if direction == 'LONG' and check_hi >= tp:
                pnl = (tp - entry_price) / entry_price * 100 - COMMISSION_PCT
                return {'reason': 'TP', 'pnl': pnl, 'bar': bar_idx}
            if direction == 'SHORT' and check_lo <= tp:
                pnl = (entry_price - tp) / entry_price * 100 - COMMISSION_PCT
                return {'reason': 'TP', 'pnl': pnl, 'bar': bar_idx}

            # SL
            if direction == 'LONG' and check_lo <= current_sl:
                r = 'TRAIL_SL' if trailing_active else 'SL'
                pnl = (current_sl - entry_price) / entry_price * 100 - COMMISSION_PCT
                return {'reason': r, 'pnl': pnl, 'bar': bar_idx}
            if direction == 'SHORT' and check_hi >= current_sl:
                r = 'TRAIL_SL' if trailing_active else 'SL'
                pnl = (entry_price - current_sl) / entry_price * 100 - COMMISSION_PCT
                return {'reason': r, 'pnl': pnl, 'bar': bar_idx}

            # Timeout
            if bar_idx >= timeout:
                if direction == 'LONG':
                    pnl = (cl - entry_price) / entry_price * 100 - COMMISSION_PCT
                else:
                    pnl = (entry_price - cl) / entry_price * 100 - COMMISSION_PCT
                return {'reason': 'TIMEOUT', 'pnl': pnl, 'bar': bar_idx}

        # Still open at end of history
        if history:
            cl = history[-1].get('close', entry_price)
            if direction == 'LONG':
                pnl = (cl - entry_price) / entry_price * 100 - COMMISSION_PCT
            else:
                pnl = (entry_price - cl) / entry_price * 100 - COMMISSION_PCT
            return {'reason': 'OPEN', 'pnl': pnl, 'bar': len(history)}

        return None

    def _grid_check(self, direction, entry_price, entry_atr, history):
        """Run all grid parameter combos on one trade's history."""
        if not history:
            return

        keys = list(GRID_PARAMS.keys())
        values = list(GRID_PARAMS.values())

        for combo in product(*values):
            params = dict(zip(keys, combo))

            trail_val = params['trail_activation']
            cfg = {
                'sl_atr': params['sl_atr'],
                'tp_atr': params['tp_atr'],
                'timeout': params['timeout'],
            }
            if trail_val == 0:
                cfg['trail_enabled'] = False
            else:
                cfg['trail_activation'] = trail_val

            alt = self._calc_alt_exit(cfg, direction, entry_price, entry_atr, history)
            if alt and alt['reason'] != 'OPEN':
                grid_name = (f"s{params['sl_atr']}_t{params['tp_atr']}"
                             f"_tr{trail_val}_to{params['timeout']}")
                r = self._grid_results[grid_name]
                r['trades'] += 1
                if alt['pnl'] > 0:
                    r['wins'] += 1
                r['pnl'] += alt['pnl']
                if direction == 'LONG':
                    r['long_n'] += 1
                    r['long_pnl'] += alt['pnl']
                else:
                    r['short_n'] += 1
                    r['short_pnl'] += alt['pnl']

    def tick(self):
        """Called once per scan cycle. Returns grid summary lines if interval reached."""
        self._tick += 1
        if self._tick % self._summary_interval == 0:
            return self._grid_summary()
        return []

    def _grid_summary(self):
        results = []
        for name, r in self._grid_results.items():
            if r['trades'] > 0:
                results.append((name, r))

        if not results:
            return [f"GRID_SUMMARY | tick={self._tick} | No closed trades yet"]

        results.sort(key=lambda x: x[1]['pnl'], reverse=True)

        lines = [f"GRID_SUMMARY | {len(results)} param combos | tick={self._tick} | top-20:"]
        for i, (name, r) in enumerate(results[:20], 1):
            wr = f"{r['wins']}/{r['trades']}={r['wins']/r['trades']*100:.0f}%" if r['trades'] else "0"
            lines.append(
                f"  GRID #{i:>2} {name} | {r['trades']}t WR={wr} PnL={r['pnl']:+.2f}% | "
                f"L:{r['long_n']}({r['long_pnl']:+.2f}%) S:{r['short_n']}({r['short_pnl']:+.2f}%)"
            )

        if len(results) > 20:
            lines.append("  --- worst-5: ---")
            for name, r in results[-5:]:
                wr = f"{r['wins']}/{r['trades']}={r['wins']/r['trades']*100:.0f}%" if r['trades'] else "0"
                lines.append(
                    f"  GRID_WORST {name} | {r['trades']}t WR={wr} PnL={r['pnl']:+.2f}%"
                )

        return lines
