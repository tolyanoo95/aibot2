#!/usr/bin/env python3
"""
Volume Bars Trading Bot — Live/Paper
─────────────────────────────────────
Rules-based momentum strategy on volume bars.
No ML. Pure rules: momentum + guards + HTF vol trend + DCA + global lock + ADX.

Usage:
    python main_volbars.py              # run live (paper mode by default)
    python main_volbars.py --once       # single scan then exit
    python main_volbars.py --live       # real trading (needs API keys)
"""

VERSION = "1.0.0"

import argparse
import functools
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars

logging.basicConfig(level=logging.WARNING)

import functools
_print = functools.partial(print, flush=True)

class _Logger:
    def __init__(self):
        self._file = open("volbars_bot.log", "a")
    def info(self, msg):
        from datetime import datetime
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        _print(line)
        self._file.write(line + "\n")
        self._file.flush()
    def warning(self, msg): self.info(f"[WARN] {msg}")
    def error(self, msg): self.info(f"[ERROR] {msg}")
    def debug(self, msg): pass

logger = _Logger()

# ── Strategy parameters ──────────────────────────────────────
SL_MULT = 2.0
TP_MULT = 4.0
LONG_MOM = 0.30
SHORT_MOM = 0.10
ADX_MIN = 25
ATR_EXP_MAX = 1.5
MAX_OPEN = 2
MAX_DCA = 3
DCA_STEP_MULT = 1.0
COOLDOWN_BARS = 3
WARMUP_DAYS = 60
SCAN_INTERVAL = 900  # 15 minutes


@dataclass
class Position:
    symbol: str
    direction: str
    entries: list = field(default_factory=list)
    avg_price: float = 0.0
    total_size: int = 0
    hard_sl: float = 0.0
    tp: float = 0.0
    entry_time: float = 0.0
    bars_held: int = 0


class VolumeBarsBot:
    def __init__(self, paper: bool = True):
        import threading
        self.paper = paper
        self.fetcher = BinanceDataFetcher(config)
        self.indicators = TechnicalIndicators()
        self.pairs = list(config.TRADING_PAIRS)

        # State
        self.time_data: Dict[str, pd.DataFrame] = {}
        self.vol_bars: Dict[str, pd.DataFrame] = {}
        self.htf_vol_bars: Dict[str, pd.DataFrame] = {}
        self.vol_buffers: Dict[str, dict] = {}
        self.vol_thresholds: Dict[str, float] = {}
        self.positions: List[Position] = []
        self.cooldowns: Dict[str, int] = {}
        self.global_locked_dir: Optional[str] = None
        self.global_sl_streak = {"LONG": 0, "SHORT": 0}
        self.scan_count = 0

        # Thread lock for shared resources (positions, global_lock, cooldowns)
        self._lock = threading.Lock()

    def initialize(self):
        """Load historical data and build volume bars."""
        logger.info(f"Initializing with {WARMUP_DAYS} days of data...")
        total_candles = WARMUP_DAYS * 96

        for symbol in self.pairs:
            logger.info(f"  Fetching {symbol}...")
            df = self.fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
            if df.empty or len(df) < 500:
                logger.warning(f"  {symbol}: not enough data, skipping")
                continue

            tdf = self.indicators.calculate_all(df.copy())
            self.time_data[symbol] = tdf

            vdf = resample_to_volume_bars(df)
            if len(vdf) < 200:
                logger.warning(f"  {symbol}: only {len(vdf)} vol bars, skipping")
                continue

            # HTF volume bars (5x threshold)
            warmup = min(1920, len(df))
            htf_threshold = df["volume"].iloc[:warmup].median() * 10
            htf_vdf = resample_to_volume_bars(df, initial_threshold=htf_threshold)
            if len(htf_vdf) > 20:
                htf_vdf = self.indicators.calculate_all(htf_vdf)
                for col in ["ema_9", "ema_21", "ema_50"]:
                    if col in htf_vdf.columns:
                        vdf[f"htf_{col}"] = htf_vdf[col].reindex(vdf.index, method="ffill")
                self.htf_vol_bars[symbol] = htf_vdf

            vdf = self.indicators.calculate_all(vdf)
            time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
            vdf["atr"] = time_atr.values

            self.vol_bars[symbol] = vdf
            self.vol_thresholds[symbol] = df["volume"].iloc[:warmup].median() * 2

            # Init volume buffer for incremental updates
            self.vol_buffers[symbol] = {
                "cum_vol": 0, "bar_open": None, "bar_high": None,
                "bar_low": None, "bar_start": None,
            }

            logger.info(f"  {symbol}: {len(vdf)} vol bars, threshold={self.vol_thresholds[symbol]:.0f}")

        logger.info(f"Initialized {len(self.vol_bars)} pairs")

    def update_volume_bars(self, symbol: str) -> bool:
        """Fetch latest 15m bar and update volume bars. Returns True if new vol bar closed."""
        try:
            df_new = self.fetcher.fetch_ohlcv(symbol, "15m", limit=2)
            if df_new.empty:
                return False

            latest = df_new.iloc[-1]
            buf = self.vol_buffers[symbol]

            # Update time data
            if symbol in self.time_data:
                self.time_data[symbol] = pd.concat([self.time_data[symbol], df_new.iloc[[-1]]])
                self.time_data[symbol] = self.time_data[symbol][~self.time_data[symbol].index.duplicated(keep='last')]
                self.time_data[symbol] = self.indicators.calculate_all(self.time_data[symbol].tail(500))

            if buf["bar_open"] is None:
                buf["bar_open"] = latest["open"]
                buf["bar_high"] = latest["high"]
                buf["bar_low"] = latest["low"]
                buf["bar_start"] = latest.name

            buf["bar_high"] = max(buf["bar_high"], latest["high"])
            buf["bar_low"] = min(buf["bar_low"], latest["low"])
            buf["cum_vol"] += latest["volume"]

            threshold = self.vol_thresholds.get(symbol, 2000)

            if buf["cum_vol"] >= threshold:
                new_bar = pd.DataFrame([{
                    "open": buf["bar_open"],
                    "high": buf["bar_high"],
                    "low": buf["bar_low"],
                    "close": latest["close"],
                    "volume": buf["cum_vol"],
                }], index=[buf["bar_start"]])

                self.vol_bars[symbol] = pd.concat([self.vol_bars[symbol], new_bar])
                self.vol_bars[symbol] = self.vol_bars[symbol][~self.vol_bars[symbol].index.duplicated(keep='last')]
                self.vol_bars[symbol] = self.indicators.calculate_all(
                    self.vol_bars[symbol].tail(500)
                )

                # Update ATR from time bars
                if symbol in self.time_data:
                    time_atr = self.time_data[symbol]["atr"].reindex(
                        self.vol_bars[symbol].index, method="ffill"
                    )
                    self.vol_bars[symbol]["atr"] = time_atr.values

                # Reset buffer
                buf["cum_vol"] = 0
                buf["bar_open"] = None

                logger.debug(f"{symbol}: new volume bar closed")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")
            return False

    def check_signal(self, symbol: str) -> Optional[dict]:
        """Check if there's a trading signal on the latest volume bar."""
        vdf = self.vol_bars.get(symbol)
        if vdf is None or len(vdf) < 50:
            return None

        j = len(vdf) - 1
        roc_12 = float(vdf["roc_12"].iloc[j]) if "roc_12" in vdf.columns else 0

        if roc_12 > LONG_MOM:
            direction = "LONG"
        elif roc_12 < -SHORT_MOM:
            direction = "SHORT"
        else:
            return None

        # Global direction lock
        if self.global_locked_dir == direction:
            return None

        # HTF volume bars trend filter
        htf_e9 = float(vdf["htf_ema_9"].iloc[j]) if "htf_ema_9" in vdf.columns else 0
        htf_e21 = float(vdf["htf_ema_21"].iloc[j]) if "htf_ema_21" in vdf.columns else 0
        if htf_e9 > 0 and htf_e21 > 0:
            if direction == "LONG" and htf_e9 < htf_e21:
                return None
            if direction == "SHORT" and htf_e9 > htf_e21:
                return None

        # ATR expansion filter
        atr = vdf["atr"].values
        atr_ma20 = pd.Series(atr).rolling(20, min_periods=1).mean().values
        atr_exp = atr[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
        if atr_exp > ATR_EXP_MAX:
            return None

        # ADX guard
        adx_col = vdf["ADX_14"]
        if isinstance(adx_col, pd.DataFrame):
            adx_col = adx_col.iloc[:, 0]
        adx = float(adx_col.iloc[j]) if "ADX_14" in vdf.columns else 25
        if adx < ADX_MIN:
            return None

        # RSI slope guard
        rsi_s6 = float(vdf["rsi"].diff(6).iloc[j]) if "rsi" in vdf.columns else 0
        if direction == "LONG" and rsi_s6 < 0:
            return None
        if direction == "SHORT" and rsi_s6 > 0:
            return None

        price = float(vdf["close"].iloc[j])
        atr_val = float(atr[j])

        return {
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "atr": atr_val,
            "adx": adx,
            "roc_12": roc_12,
        }

    def open_position(self, signal: dict):
        """Open a new position or add DCA entry."""
        symbol = signal["symbol"]
        direction = signal["direction"]
        price = signal["price"]
        atr_val = signal["atr"]

        # Check cooldown
        if self.cooldowns.get(symbol, 0) > self.scan_count:
            return

        # Check max open
        if len(self.positions) >= MAX_OPEN:
            return

        # Check if already have position on this symbol
        existing = [p for p in self.positions if p.symbol == symbol]
        if existing:
            pos = existing[0]
            if pos.direction != direction:
                return
            if pos.total_size >= MAX_DCA:
                return

            # DCA: add entry
            dca_level = pos.entries[0][0] - pos.total_size * DCA_STEP_MULT * atr_val if direction == "LONG" \
                else pos.entries[0][0] + pos.total_size * DCA_STEP_MULT * atr_val

            if (direction == "LONG" and price <= dca_level) or (direction == "SHORT" and price >= dca_level):
                pos.entries.append((price, time.time()))
                pos.total_size += 1
                pos.avg_price = sum(e[0] for e in pos.entries) / pos.total_size
                pos.tp = pos.avg_price + TP_MULT * atr_val if direction == "LONG" else pos.avg_price - TP_MULT * atr_val
                logger.info(f"  DCA #{pos.total_size} {symbol} {direction} @ {price:.2f} (avg: {pos.avg_price:.2f})")

                if self.paper:
                    logger.info(f"  [PAPER] DCA entry {symbol}")
            return

        # New position
        if direction == "LONG":
            hard_sl = price - SL_MULT * atr_val
            tp = price + TP_MULT * atr_val
        else:
            hard_sl = price + SL_MULT * atr_val
            tp = price - TP_MULT * atr_val

        pos = Position(
            symbol=symbol, direction=direction,
            entries=[(price, time.time())],
            avg_price=price, total_size=1,
            hard_sl=hard_sl, tp=tp,
            entry_time=time.time(),
        )
        self.positions.append(pos)

        logger.info(f"  OPEN {direction} {symbol} @ {price:.2f} | SL={hard_sl:.2f} TP={tp:.2f} ATR={atr_val:.2f}")
        if self.paper:
            logger.info(f"  [PAPER] Position opened")

    # Old check_positions removed — replaced by _check_pair_positions (per-pair, thread-safe)

    def _process_pair(self, symbol: str):
        """Fully independent pair processing: scan + signal + open + close. Runs in own thread."""
        try:
            # 1. Update volume bars (pair-specific data, no lock needed)
            new_bar = self.update_volume_bars(symbol)
            if not new_bar:
                # Still check existing positions for this pair
                self._check_pair_positions(symbol)
                return

            # 2. Check signal (reads shared global_locked_dir but doesn't modify)
            signal = self.check_signal(symbol)

            # 3. Open position if signal (needs lock for shared state)
            if signal:
                logger.info(f"  SIGNAL: {signal['direction']} {signal['symbol']} roc={signal['roc_12']:.2f}% ADX={signal['adx']:.0f}")
                with self._lock:
                    self.open_position(signal)

            # 4. Check positions for this pair (needs lock)
            self._check_pair_positions(symbol)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    def _check_pair_positions(self, symbol: str):
        """Check SL/TP/timeout for positions of this specific pair."""
        with self._lock:
            for pos in list(self.positions):
                if pos.symbol != symbol:
                    continue

                vdf = self.vol_bars.get(pos.symbol)
                if vdf is None or len(vdf) < 2:
                    continue

                current_price = float(vdf["close"].iloc[-1])
                current_high = float(vdf["high"].iloc[-1])
                current_low = float(vdf["low"].iloc[-1])
                pos.bars_held += 1

                hit_tp = hit_sl = False
                if pos.direction == "LONG":
                    hit_tp = current_high >= pos.tp
                    hit_sl = current_low <= pos.hard_sl
                else:
                    hit_tp = current_low <= pos.tp
                    hit_sl = current_high >= pos.hard_sl

                exit_reason = None
                exit_price = current_price

                if hit_sl and hit_tp:
                    exit_reason = "HARD_SL"
                    exit_price = pos.hard_sl
                elif hit_sl:
                    exit_reason = "HARD_SL"
                    exit_price = pos.hard_sl
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = pos.tp
                elif pos.bars_held >= 24:
                    exit_reason = "TIMEOUT"
                    exit_price = current_price

                if exit_reason:
                    if pos.direction == "LONG":
                        pnl_pct = (exit_price - pos.avg_price) / pos.avg_price * 100 * pos.total_size
                    else:
                        pnl_pct = (pos.avg_price - exit_price) / pos.avg_price * 100 * pos.total_size

                    logger.info(
                        f"  CLOSE {pos.direction} {pos.symbol} @ {exit_price:.2f} "
                        f"| {exit_reason} | PnL {pnl_pct:+.2f}% | Bars: {pos.bars_held} | DCA: {pos.total_size}"
                    )

                    # Update global direction lock
                    is_dca_sl = exit_reason == "HARD_SL" and pos.total_size > 1
                    if is_dca_sl:
                        self.global_locked_dir = pos.direction
                        logger.info(f"  LOCK {pos.direction} (DCA SL)")
                    elif exit_reason == "HARD_SL":
                        self.global_sl_streak[pos.direction] += 1
                        if self.global_sl_streak[pos.direction] >= 2:
                            self.global_locked_dir = pos.direction
                            logger.info(f"  LOCK {pos.direction} (2 SL streak)")
                    elif exit_reason == "TP":
                        self.global_sl_streak[pos.direction] = 0
                        if self.global_locked_dir == pos.direction:
                            self.global_locked_dir = None
                            logger.info(f"  UNLOCK {pos.direction}")

                    self.positions.remove(pos)
                    self.cooldowns[pos.symbol] = self.scan_count + COOLDOWN_BARS

    def scan(self):
        """Run one scan cycle — each pair fully independent in its own thread."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.scan_count += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"Scan #{self.scan_count} | Positions: {len(self.positions)} | Lock: {self.global_locked_dir or 'none'}")

        symbols = list(self.vol_bars.keys())

        with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
            futures = {pool.submit(self._process_pair, sym): sym for sym in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error {sym}: {e}")

        # Status
        for pos in self.positions:
            vdf = self.vol_bars.get(pos.symbol)
            if vdf is not None:
                current = float(vdf["close"].iloc[-1])
                if pos.direction == "LONG":
                    unrealized = (current - pos.avg_price) / pos.avg_price * 100
                else:
                    unrealized = (pos.avg_price - current) / pos.avg_price * 100
                logger.info(
                    f"  HOLDING: {pos.direction} {pos.symbol} entry={pos.avg_price:.2f} "
                    f"now={current:.2f} PnL={unrealized:+.2f}% bars={pos.bars_held} dca={pos.total_size}"
                )

        logger.info(f"  Positions: {len(self.positions)} | Total pairs: {len(self.vol_bars)}")

    def run(self, once: bool = False):
        """Main loop."""
        logger.info(f"Volume Bars Bot v{VERSION}")
        logger.info(f"Mode: {'PAPER' if self.paper else 'LIVE'}")
        logger.info(f"Params: SL={SL_MULT}x TP={TP_MULT}x ADX>{ADX_MIN} MaxOpen={MAX_OPEN} DCA={MAX_DCA}")
        logger.info(f"Pairs: {len(self.pairs)}")

        self.initialize()

        if once:
            self.scan()
            return

        logger.info(f"Starting scan loop (every {SCAN_INTERVAL}s)...")
        while True:
            try:
                self.scan()
                logger.info(f"  Next scan in {SCAN_INTERVAL}s...")
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume Bars Trading Bot")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    parser.add_argument("--live", action="store_true", help="Real trading (default: paper)")
    args = parser.parse_args()

    bot = VolumeBarsBot(paper=not args.live)
    bot.run(once=args.once)
