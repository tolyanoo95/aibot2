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

    def __init__(self, name: str, config: dict, base_config):
        self.name = name
        self._cfg = config
        self._base = base_config
        self.open_trades: dict[str, ShadowTrade] = {}
        self.pair_cooldown: dict[str, int] = {}
        self.pair_sl_cooldown: dict[str, int] = {}
        self.consecutive_sl: int = 0
        self.global_pause: int = 0
        self._tick: int = 0
        self._log_buffer: list[str] = []

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

        # TIMEOUT
        if trade.bars_held >= self.timeout:
            if trade.direction == 'LONG':
                pnl = (sig.close - trade.entry_price) / trade.entry_price * 100 - COMMISSION_PCT
            else:
                pnl = (trade.entry_price - sig.close) / trade.entry_price * 100 - COMMISSION_PCT
            self._close_trade(trade, 'TIMEOUT', pnl)

    def _try_open(self, sig: SignalData, threshold: float):
        if sig.ml_dir == 'NEUTRAL':
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

        if self.use_fresh and sig.ml_conf < 0.50:
            return

        eff_conf = sig.ml_conf
        if sig.ml_disagr >= 0.5:
            eff_conf *= 0.60
        elif sig.ml_disagr > 0:
            eff_conf *= 0.85

        if eff_conf < threshold:
            return

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

        self._log_buffer.append(
            f"SIM_OPEN {self.name} | {sig.symbol} {direction} | entry={sig.price:.6g} | "
            f"sl={sl:.6g} | tp={tp:.6g} | conf={eff_conf:.1%} | regime={sig.regime}"
        )

    def _close_trade(self, trade: ShadowTrade, reason: str, pnl: float):
        del self.open_trades[trade.symbol]
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
    """Manages multiple shadow portfolios."""

    def __init__(self, config_path: str = "shadow_configs.json", base_config=None):
        self.portfolios: list[ShadowPortfolio] = []
        self._base_config = base_config

        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    configs = json.load(f)
                for cfg in configs:
                    name = cfg.pop('name', f'shadow_{len(self.portfolios)}')
                    self.portfolios.append(ShadowPortfolio(name, cfg, base_config))
                logger.info("Loaded %d shadow configs from %s", len(self.portfolios), config_path)
            except Exception as exc:
                logger.error("Failed to load shadow configs: %s", exc)
        else:
            logger.info("No shadow_configs.json found — shadow trading disabled")

    def process_signals(self, all_signals: list[SignalData]):
        for portfolio in self.portfolios:
            portfolio.process_tick(all_signals)

    def flush_all_logs(self) -> list[str]:
        lines = []
        for p in self.portfolios:
            lines.extend(p.flush_logs())
        return lines

    def get_all_summaries(self) -> list[dict]:
        return [p.get_summary() for p in self.portfolios]

    @property
    def active(self) -> bool:
        return len(self.portfolios) > 0
