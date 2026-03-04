"""
Shadow portfolio system: run multiple virtual configs in parallel with real trading.
Each config tracks its own trades, trailing stops, cooldowns — 100% accurate.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

COMMISSION_PCT = 0.08


@dataclass
class ShadowTrade:
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    current_sl: float
    best_price: float
    confidence: float
    regime: str
    open_time: datetime
    bars_held: int = 0
    trailing_active: bool = False
    health: str = "HEALTHY"


@dataclass
class SignalData:
    symbol: str
    ml_dir: str          # LONG / SHORT / NEUTRAL
    ml_conf: float       # 0-1 raw confidence
    ml_disagr: float     # 0-1 disagreement
    regime: str
    price: float
    high: float
    low: float
    close: float
    atr: float
    ml_signal: int = 0   # 1=BUY, -1=SELL, 0=HOLD
    candles_1m: list = field(default_factory=list)


class ShadowPortfolio:
    """One virtual portfolio with its own config and full trade state."""

    def __init__(self, name: str, config: dict, base_config, verbose: bool = True):
        self.name = name
        self._cfg = config
        self._base = base_config
        self._verbose = verbose  # if False, don't log individual trades (grid mode)
        self.open_trades: dict[str, ShadowTrade] = {}
        self.pair_cooldown: dict[str, int] = {}
        self.pair_sl_cooldown: dict[str, int] = {}
        self.consecutive_sl: int = 0
        self.global_pause: int = 0
        self._tick: int = 0
        self._log_buffer: list[str] = []
        self._active_signals: dict[str, str] = {}  # sym → direction (signal tracking)

        self.total_trades = 0
        self.total_wins = 0
        self.total_pnl = 0.0
        self.long_count = 0
        self.long_pnl = 0.0
        self.short_count = 0
        self.short_pnl = 0.0
        self._last_summary_tick = 0

    def _get(self, key: str, default=None):
        if key in self._cfg:
            return self._cfg[key]
        if default is not None:
            return default
        attr_map = {
            'threshold': 'PREDICTION_THRESHOLD',
            'sl_atr': 'SL_ATR_MULTIPLIER',
            'tp_atr': 'TP_ATR_MULTIPLIER',
            'trail_activation': 'TRAILING_ACTIVATION_ATR',
            'trail_distance': 'TRAILING_DISTANCE_ATR',
            'max_open': 'MAX_OPEN_TRADES',
            'timeout': 'MAX_HOLD_BARS',
        }
        if key in attr_map:
            return getattr(self._base, attr_map[key])
        return default

    @property
    def threshold(self):
        return self._get('threshold')

    @property
    def max_open(self):
        return self._get('max_open')

    @property
    def timeout(self):
        return self._get('timeout')

    @property
    def cooldown(self):
        return self._get('cooldown', 4)

    @property
    def trail_enabled(self):
        return self._get('trail_enabled', True)

    @property
    def trail_activation(self):
        return self._get('trail_activation')

    @property
    def trail_distance(self):
        return self._get('trail_distance')

    @property
    def inverse(self):
        return self._get('inverse', False)

    @property
    def use_fresh(self):
        return self._get('use_fresh', False)

    @property
    def exit_mode(self):
        return self._get('exit_mode', 'high_low')

    def _dead_hours(self):
        dh = self._cfg.get('dead_hours')
        if dh is not None:
            return dh
        return getattr(self._base, 'FILTER_DEAD_HOURS', [])

    def _skip_regimes(self):
        sr = self._cfg.get('skip_regimes', [])
        return [tuple(x) for x in sr]

    def _sl_tp_for_regime(self, regime: str, atr: float):
        regime_sl = self._cfg.get('regime_sl', {})
        regime_tp = self._cfg.get('regime_tp', {})

        sl_mult = regime_sl.get(regime, self._get('sl_atr'))
        tp_mult = regime_tp.get(regime, self._get('tp_atr'))

        return atr * sl_mult, atr * tp_mult

    def process_tick(self, all_signals: list[SignalData]):
        self._tick += 1

        if self.global_pause > 0:
            self.global_pause -= 1

        for sym, cd in list(self.pair_cooldown.items()):
            self.pair_cooldown[sym] -= 1
            if self.pair_cooldown[sym] <= 0:
                del self.pair_cooldown[sym]
        for sym, cd in list(self.pair_sl_cooldown.items()):
            self.pair_sl_cooldown[sym] -= 1
            if self.pair_sl_cooldown[sym] <= 0:
                del self.pair_sl_cooldown[sym]

        # Update existing trades
        for sig in all_signals:
            if sig.symbol in self.open_trades:
                self._update_trade(sig)

        # Try open new trades
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self._dead_hours():
            return

        if self.global_pause > 0:
            return

        threshold = self.threshold
        if self.consecutive_sl >= 3:
            threshold += 0.15
        elif self.consecutive_sl >= 2:
            threshold += 0.075

        for sig in all_signals:
            if len(self.open_trades) >= self.max_open:
                break
            self._try_open(sig, threshold)

        # Summary
        if self.total_trades > 0 and self.total_trades % 50 == 0 and self.total_trades != self._last_summary_tick:
            self._last_summary_tick = self.total_trades
            self._emit_summary()

    def _update_trade(self, sig: SignalData):
        trade = self.open_trades[sig.symbol]

        prices_to_check = []
        if sig.candles_1m:
            for c in sig.candles_1m:
                prices_to_check.append((c.get('high', sig.high), c.get('low', sig.low), c.get('close', sig.close)))

        prices_to_check.append((sig.high, sig.low, sig.close))

        for hi, lo, cl in prices_to_check:
            if trade.direction == 'LONG':
                trade.best_price = max(trade.best_price, hi)
            else:
                trade.best_price = min(trade.best_price, lo)

            # Trailing SL
            if self.trail_enabled and sig.atr > 0:
                act_dist = sig.atr * self.trail_activation
                tr_dist = sig.atr * self.trail_distance
                if trade.direction == 'LONG' and trade.best_price >= trade.entry_price + act_dist:
                    trade.trailing_active = True
                    new_sl = trade.best_price - tr_dist
                    if new_sl > trade.current_sl:
                        trade.current_sl = new_sl
                elif trade.direction == 'SHORT' and trade.best_price <= trade.entry_price - act_dist:
                    trade.trailing_active = True
                    new_sl = trade.best_price + tr_dist
                    if new_sl < trade.current_sl:
                        trade.current_sl = new_sl

            check_hi = hi if self.exit_mode == 'high_low' else cl
            check_lo = lo if self.exit_mode == 'high_low' else cl

            # TP
            if trade.direction == 'LONG' and check_hi >= trade.tp:
                pnl = (trade.tp - trade.entry_price) / trade.entry_price * 100 - COMMISSION_PCT
                self._close_trade(trade, 'TP', pnl)
                return
            if trade.direction == 'SHORT' and check_lo <= trade.tp:
                pnl = (trade.entry_price - trade.tp) / trade.entry_price * 100 - COMMISSION_PCT
                self._close_trade(trade, 'TP', pnl)
                return

            # SL
            if trade.direction == 'LONG' and check_lo <= trade.current_sl:
                reason = 'TRAIL_SL' if trade.trailing_active else 'SL'
                pnl = (trade.current_sl - trade.entry_price) / trade.entry_price * 100 - COMMISSION_PCT
                self._close_trade(trade, reason, pnl)
                return
            if trade.direction == 'SHORT' and check_hi >= trade.current_sl:
                reason = 'TRAIL_SL' if trade.trailing_active else 'SL'
                pnl = (trade.entry_price - trade.current_sl) / trade.entry_price * 100 - COMMISSION_PCT
                self._close_trade(trade, reason, pnl)
                return

        trade.bars_held += 1

        # EARLY_EXIT: ML reversed with strong confidence (matches real bot logic)
        if sig.ml_conf >= 0.65 and trade.bars_held >= 3:
            ml_reversed = False
            if trade.direction == 'LONG' and sig.ml_signal == -1:
                ml_reversed = True
            elif trade.direction == 'SHORT' and sig.ml_signal == 1:
                ml_reversed = True
            if ml_reversed:
                trade.health = "CLOSE_EARLY"
                if trade.direction == 'LONG':
                    pnl = (sig.close - trade.entry_price) / trade.entry_price * 100 - COMMISSION_PCT
                else:
                    pnl = (trade.entry_price - sig.close) / trade.entry_price * 100 - COMMISSION_PCT
                self._close_trade(trade, 'EARLY_EXIT', pnl)
                return

        # TIMEOUT
        if trade.bars_held >= self.timeout:
            if trade.direction == 'LONG':
                pnl = (sig.close - trade.entry_price) / trade.entry_price * 100 - COMMISSION_PCT
            else:
                pnl = (trade.entry_price - sig.close) / trade.entry_price * 100 - COMMISSION_PCT
            self._close_trade(trade, 'TIMEOUT', pnl)

    def _try_open(self, sig: SignalData, threshold: float):
        if sig.ml_dir == 'NEUTRAL':
            # Signal gone — clear tracking
            self._active_signals.pop(sig.symbol, None)
            return

        direction = sig.ml_dir
        if self.inverse:
            direction = 'SHORT' if direction == 'LONG' else 'LONG'

        if sig.symbol in self.open_trades:
            return
        if sig.symbol in self.pair_cooldown:
            return
        if sig.symbol in self.pair_sl_cooldown:
            return

        if (direction, sig.regime) in self._skip_regimes():
            return

        eff_conf = sig.ml_conf
        if sig.ml_disagr >= 0.5:
            eff_conf *= 0.60
        elif sig.ml_disagr > 0:
            eff_conf *= 0.85

        if eff_conf < threshold:
            return

        # Signal tracking: only open on NEW signals, not repeated ones
        prev = self._active_signals.get(sig.symbol)
        if prev == direction:
            return  # same signal already seen, don't re-open
        self._active_signals[sig.symbol] = direction

        sl_dist, tp_dist = self._sl_tp_for_regime(sig.regime, sig.atr)

        min_sl = sig.price * 0.3 / 100
        if sl_dist < min_sl:
            sl_dist = min_sl

        if direction == 'LONG':
            sl = sig.price - sl_dist
            tp = sig.price + tp_dist
        else:
            sl = sig.price + sl_dist
            tp = sig.price - tp_dist

        trade = ShadowTrade(
            symbol=sig.symbol,
            direction=direction,
            entry_price=sig.price,
            sl=sl,
            tp=tp,
            current_sl=sl,
            best_price=sig.price,
            confidence=eff_conf,
            regime=sig.regime,
            open_time=datetime.now(timezone.utc),
        )
        self.open_trades[sig.symbol] = trade

        if self._verbose:
            self._log_buffer.append(
                f"SIM_OPEN {self.name} | {sig.symbol} {direction} | entry={sig.price:.6g} | "
                f"sl={sl:.6g} | tp={tp:.6g} | conf={eff_conf:.1%} | regime={sig.regime}"
            )

    def _close_trade(self, trade: ShadowTrade, reason: str, pnl: float):
        del self.open_trades[trade.symbol]
        self._active_signals.pop(trade.symbol, None)
        self.pair_cooldown[trade.symbol] = self.cooldown

        is_loss = reason == 'SL' or pnl < -0.2
        if is_loss:
            self.consecutive_sl += 1
            self.pair_sl_cooldown[trade.symbol] = 8
            if self.consecutive_sl >= 3:
                self.global_pause = 24
        else:
            self.consecutive_sl = 0

        self.total_trades += 1
        if pnl > 0:
            self.total_wins += 1
        self.total_pnl += pnl
        if trade.direction == 'LONG':
            self.long_count += 1
            self.long_pnl += pnl
        else:
            self.short_count += 1
            self.short_pnl += pnl

        if self._verbose:
            self._log_buffer.append(
                f"SIM_CLOSE {self.name} | {trade.symbol} {trade.direction} | "
                f"entry={trade.entry_price:.6g} | exit={trade.current_sl if 'SL' in reason else trade.tp if reason == 'TP' else trade.entry_price:.6g} | "
                f"reason={reason} | pnl={pnl:+.2f}% | bars={trade.bars_held} | conf={trade.confidence:.1%}"
            )

    def _emit_summary(self):
        wr = f"{self.total_wins}/{self.total_trades}={self.total_wins / self.total_trades * 100:.0f}%" if self.total_trades else "0"
        self._log_buffer.append(
            f"SIM_SUMMARY {self.name} | trades={self.total_trades} | wr={wr} | "
            f"pnl={self.total_pnl:+.2f}% | "
            f"L:{self.long_count}({self.long_pnl:+.2f}%) S:{self.short_count}({self.short_pnl:+.2f}%)"
        )

    def flush_logs(self) -> list[str]:
        lines = self._log_buffer
        self._log_buffer = []
        return lines

    def get_summary(self) -> dict:
        return {
            'name': self.name,
            'trades': self.total_trades,
            'wins': self.total_wins,
            'wr': self.total_wins / self.total_trades * 100 if self.total_trades else 0,
            'pnl': self.total_pnl,
            'long_count': self.long_count,
            'long_pnl': self.long_pnl,
            'short_count': self.short_count,
            'short_pnl': self.short_pnl,
            'open': len(self.open_trades),
        }


class ShadowManager:
    """Manages multiple shadow portfolios (manual configs + grid search)."""

    GRID_PARAMS = {
        'threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
        'sl_atr': [1.0, 1.5, 2.0],
        'tp_atr': [1.5, 2.0, 2.5, 3.0],
        'trail_activation': [1.5, 2.0, 2.5, 0],  # 0 = disabled
        'timeout': [12, 18, 24],
        'max_open': [2, 4, 11],
        'dead_hours_on': [True, False],
    }

    def __init__(self, config_path: str = "shadow_configs.json", base_config=None,
                 enable_grid: bool = True):
        self.portfolios: list[ShadowPortfolio] = []
        self.grid_portfolios: list[ShadowPortfolio] = []
        self._base_config = base_config
        self._summary_interval = 12  # emit grid summary every N ticks (~1 hour)
        self._tick_count = 0

        # Load manual configs
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    configs = json.load(f)
                for cfg in configs:
                    name = cfg.pop('name', f'shadow_{len(self.portfolios)}')
                    self.portfolios.append(ShadowPortfolio(name, cfg, base_config))
                logger.info("Loaded %d manual shadow configs", len(self.portfolios))
            except Exception as exc:
                logger.error("Failed to load shadow configs: %s", exc)

        # Generate grid search configs
        if enable_grid:
            self._generate_grid(base_config)

    def _generate_grid(self, base_config):
        from itertools import product

        keys = list(self.GRID_PARAMS.keys())
        values = list(self.GRID_PARAMS.values())

        for combo in product(*values):
            params = dict(zip(keys, combo))

            trail_val = params.pop('trail_activation')
            dead_on = params.pop('dead_hours_on')

            cfg = {
                'threshold': params['threshold'],
                'sl_atr': params['sl_atr'],
                'tp_atr': params['tp_atr'],
                'timeout': params['timeout'],
                'max_open': params['max_open'],
            }

            if trail_val == 0:
                cfg['trail_enabled'] = False
            else:
                cfg['trail_activation'] = trail_val

            if not dead_on:
                cfg['dead_hours'] = []

            name = (f"g_c{params['threshold']:.0%}_s{params['sl_atr']}_t{params['tp_atr']}"
                    f"_tr{trail_val}_to{params['timeout']}_m{params['max_open']}"
                    f"_d{'Y' if dead_on else 'N'}")

            self.grid_portfolios.append(ShadowPortfolio(name, cfg, base_config, verbose=False))

        logger.info("Generated %d grid search portfolios", len(self.grid_portfolios))

    def process_signals(self, all_signals: list[SignalData]):
        for p in self.portfolios:
            p.process_tick(all_signals)
        for p in self.grid_portfolios:
            p.process_tick(all_signals)
        self._tick_count += 1

    def flush_all_logs(self) -> list[str]:
        lines = []

        # Manual configs: full logging (SIM_OPEN, SIM_CLOSE)
        for p in self.portfolios:
            lines.extend(p.flush_logs())

        # Grid configs: discard individual trade logs (too many)
        for p in self.grid_portfolios:
            p.flush_logs()

        # Grid summary: emit top-20 every N ticks
        if self._tick_count > 0 and self._tick_count % self._summary_interval == 0:
            lines.extend(self._grid_summary())

        return lines

    def _grid_summary(self) -> list[str]:
        results = []
        for p in self.grid_portfolios:
            s = p.get_summary()
            if s['trades'] > 0:
                results.append(s)

        if not results:
            return ["GRID_SUMMARY | No trades yet"]

        results.sort(key=lambda x: x['pnl'], reverse=True)

        lines = [
            f"GRID_SUMMARY | {len(results)} configs with trades | "
            f"tick={self._tick_count} | top-20:"
        ]
        for i, r in enumerate(results[:20], 1):
            lines.append(
                f"  GRID #{i:>2} {r['name']} | {r['trades']}t WR={r['wr']:.0f}% "
                f"PnL={r['pnl']:+.2f}% | L:{r['long_count']}({r['long_pnl']:+.2f}%) "
                f"S:{r['short_count']}({r['short_pnl']:+.2f}%)"
            )

        # Also bottom-5 (worst)
        if len(results) > 20:
            lines.append("  --- worst-5: ---")
            for r in results[-5:]:
                lines.append(
                    f"  GRID_WORST {r['name']} | {r['trades']}t WR={r['wr']:.0f}% "
                    f"PnL={r['pnl']:+.2f}%"
                )

        return lines

    def get_all_summaries(self) -> list[dict]:
        all_s = [p.get_summary() for p in self.portfolios]
        all_s.extend(p.get_summary() for p in self.grid_portfolios)
        return all_s

    def get_grid_top(self, n: int = 20) -> list[dict]:
        results = [p.get_summary() for p in self.grid_portfolios if p.total_trades > 0]
        results.sort(key=lambda x: x['pnl'], reverse=True)
        return results[:n]

    @property
    def active(self) -> bool:
        return len(self.portfolios) > 0 or len(self.grid_portfolios) > 0
