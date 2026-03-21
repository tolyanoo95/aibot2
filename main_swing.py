#!/usr/bin/env python3
"""
Swing Bot — 4h+15m Supertrend + MFI/RSI/ATR filters
────────────────────────────────────────────────────
Entry: LTF 15m Supertrend aligns with HTF 4h Supertrend + 4 filters
Exit: HTF 4h Supertrend flips
On startup: wait for first flip (don't enter immediately)

Backtest: +$3,242/mo, 37/40 months profitable (Dec 2022 — Mar 2026).

Usage:
    python main_swing.py               # paper mode
    python main_swing.py --once        # show current state and exit
"""

VERSION = "0.2.0"

import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as pta

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.indicators import TechnicalIndicators

logger = logging.getLogger("swing")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)
_fh = logging.FileHandler("swing_bot.log")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

WARMUP_DAYS = 60
SCAN_INTERVAL = 900  # 15 minutes


def _pfmt(price: float) -> str:
    ap = abs(price)
    if ap >= 1000: return f"{price:.2f}"
    if ap >= 10: return f"{price:.2f}"
    if ap >= 1: return f"{price:.4f}"
    return f"{price:.6f}"


class SwingBot:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.fetcher = BinanceDataFetcher(config)
        self.pairs = list(config.TRADING_PAIRS)

        # Per-pair state
        self.data_15m: Dict[str, pd.DataFrame] = {}
        self.htf_st_dir: Dict[str, float] = {}    # current 4h Supertrend direction
        self.ltf_st_dir: Dict[str, float] = {}    # current 15m Supertrend direction
        self.positions: Dict[str, dict] = {}       # pair -> {direction, entry_price, entry_time}
        self.waiting_first_flip: Dict[str, bool] = {}  # True until first HTF flip after startup

        self._trades_file = "swing_trades.json"
        self._state_file = "data/swing_state.json"
        self._file_lock = threading.Lock()
        os.makedirs("data", exist_ok=True)

    def _read_trades(self) -> list:
        with self._file_lock:
            if not os.path.exists(self._trades_file):
                return []
            try:
                with open(self._trades_file) as f:
                    return json.load(f)
            except Exception:
                return []

    def _write_trades(self, data):
        with self._file_lock:
            with open(self._trades_file, "w") as f:
                json.dump(data, f, indent=2)

    def _save_state(self):
        state = {
            "positions": self.positions,
            "htf_st_dir": {k: float(v) for k, v in self.htf_st_dir.items()},
            "ltf_st_dir": {k: float(v) for k, v in self.ltf_st_dir.items()},
            "waiting_first_flip": self.waiting_first_flip,
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                state = json.load(f)
            self.positions = state.get("positions", {})
            self.htf_st_dir = {k: float(v) for k, v in state.get("htf_st_dir", {}).items()}
            self.ltf_st_dir = {k: float(v) for k, v in state.get("ltf_st_dir", {}).items()}
            self.waiting_first_flip = state.get("waiting_first_flip", {})
            if self.positions:
                logger.info(f"  Restored {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"State load error: {e}")

    def _compute_pair(self, pair: str, df_15m: pd.DataFrame) -> dict:
        """Compute indicators for one pair."""
        htf = df_15m.resample("240min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        st_htf = pta.supertrend(htf["high"], htf["low"], htf["close"], length=9, multiplier=3.0)
        if st_htf is None:
            return None
        htf_st = st_htf[[c for c in st_htf.columns if "SUPERTd" in c][0]]

        st_ltf = pta.supertrend(df_15m["high"], df_15m["low"], df_15m["close"], length=9, multiplier=3.0)
        if st_ltf is None:
            return None
        ltf_st = st_ltf[[c for c in st_ltf.columns if "SUPERTd" in c][0]]

        def sr(s):
            return s.reindex(df_15m.index, method="ffill") if s is not None else None

        mfi_h = pta.mfi(htf["high"], htf["low"], htf["close"], htf["volume"], length=14)
        rsi_h = pta.rsi(htf["close"], length=14)
        atr = pta.atr(htf["high"], htf["low"], htf["close"], length=14)
        atr_ratio = (atr / atr.rolling(20).mean()) if atr is not None else None
        mfi_l = pta.mfi(df_15m["high"], df_15m["low"], df_15m["close"], df_15m["volume"], length=14)

        return {
            "htf_st_val": float(htf_st.dropna().iloc[-1]),
            "ltf_st_val": float(ltf_st.dropna().iloc[-1]),
            "ltf_st_prev": float(ltf_st.dropna().iloc[-2]) if len(ltf_st.dropna()) >= 2 else np.nan,
            "price": float(df_15m["close"].iloc[-1]),
            "H_mfi": float(sr(mfi_h).dropna().iloc[-1]) if mfi_h is not None else np.nan,
            "H_rsi": float(sr(rsi_h).dropna().iloc[-1]) if rsi_h is not None else np.nan,
            "H_atr_r": float(sr(atr_ratio).dropna().iloc[-1]) if atr_ratio is not None else np.nan,
            "L_mfi": float(mfi_l.dropna().iloc[-1]) if mfi_l is not None else np.nan,
        }

    def _check_filters(self, ind: dict, htf_dir: float) -> bool:
        """Check 4 entry filters."""
        h = htf_dir
        mfi_h = ind["H_mfi"]
        rsi_h = ind["H_rsi"]
        atr_r = ind["H_atr_r"]
        mfi_l = ind["L_mfi"]

        if np.isnan(mfi_h) or np.isnan(rsi_h) or np.isnan(atr_r) or np.isnan(mfi_l):
            return False
        if h > 0 and mfi_h <= 50: return False
        if h < 0 and mfi_h >= 50: return False
        if h > 0 and rsi_h <= 50: return False
        if h < 0 and rsi_h >= 50: return False
        if atr_r >= 1.5: return False
        if h > 0 and mfi_l <= 50: return False
        if h < 0 and mfi_l >= 50: return False
        return True

    def initialize(self):
        """Fetch data and compute initial state."""
        self._load_state()
        logger.info(f"Fetching {WARMUP_DAYS} days of 15m data...")
        total_candles = WARMUP_DAYS * 96

        for pair in self.pairs:
            logger.info(f"  Fetching {pair}...")
            df = self.fetcher.fetch_ohlcv_extended(pair, "15m", total_candles=total_candles)
            if df.empty or len(df) < 500:
                logger.warning(f"  {pair}: not enough data")
                continue
            self.data_15m[pair] = df

            ind = self._compute_pair(pair, df)
            if ind is None:
                continue

            old_htf = self.htf_st_dir.get(pair, 0)
            new_htf = ind["htf_st_val"]
            self.htf_st_dir[pair] = new_htf
            self.ltf_st_dir[pair] = ind["ltf_st_val"]

            # On first run: mark as waiting for flip
            if pair not in self.waiting_first_flip:
                if pair not in self.positions:
                    self.waiting_first_flip[pair] = True
                    logger.info(f"  {pair}: waiting for first HTF flip before trading")
                else:
                    self.waiting_first_flip[pair] = False

            st_label = "LONG" if new_htf > 0 else "SHORT"
            pos_str = f" | POS: {self.positions[pair]['direction']}" if pair in self.positions else ""
            wait_str = " | WAITING" if self.waiting_first_flip.get(pair) else ""
            logger.info(f"  {pair}: ST={st_label} MFI_h={ind['H_mfi']:.0f} RSI_h={ind['H_rsi']:.0f} ATR_r={ind['H_atr_r']:.2f}{pos_str}{wait_str}")

        logger.info(f"Initialized {len(self.data_15m)} pairs")
        self._save_state()

    def _scan_pair(self, pair: str):
        """Scan a single pair — runs in its own thread."""
        try:
            df_new = self.fetcher.fetch_ohlcv(pair, "15m", limit=5)
            if df_new.empty:
                return
            base = self.data_15m[pair][["open", "high", "low", "close", "volume"]]
            base = pd.concat([base, df_new])
            base = base[~base.index.duplicated(keep="last")].sort_index().tail(WARMUP_DAYS * 96)
            self.data_15m[pair] = base
        except Exception as e:
            logger.error(f"  {pair} fetch error: {e}")
            return

        ind = self._compute_pair(pair, self.data_15m[pair])
        if ind is None:
            return

        price = ind["price"]
        new_htf = ind["htf_st_val"]
        new_ltf = ind["ltf_st_val"]
        old_htf = self.htf_st_dir.get(pair, new_htf)
        old_ltf = self.ltf_st_dir.get(pair, new_ltf)

        # Detect HTF flip
        htf_flipped = (old_htf != 0 and new_htf != old_htf)

        if htf_flipped:
            new_dir = "LONG" if new_htf > 0 else "SHORT"
            logger.info(f"  HTF FLIP {pair} → {new_dir} @ {_pfmt(price)}")

            if self.waiting_first_flip.get(pair):
                self.waiting_first_flip[pair] = False
                logger.info(f"  {pair}: first flip detected, trading enabled")

            if pair in self.positions:
                self._close_position(pair, price, "HTF_FLIP")

        # Detect LTF flip (entry signal)
        ltf_flipped = (old_ltf != 0 and new_ltf != old_ltf)

        if ltf_flipped and pair not in self.positions:
            if not self.waiting_first_flip.get(pair):
                aligned = (new_htf > 0 and new_ltf > 0) or (new_htf < 0 and new_ltf < 0)
                if aligned and self._check_filters(ind, new_htf):
                    direction = "LONG" if new_htf > 0 else "SHORT"
                    self._open_position(pair, direction, price, ind)

        self.htf_st_dir[pair] = new_htf
        self.ltf_st_dir[pair] = new_ltf

    def scan(self):
        """Run one scan: all pairs in parallel."""
        now = datetime.now().strftime("%H:%M:%S")
        pairs = list(self.data_15m.keys())

        with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
            futures = {executor.submit(self._scan_pair, pair): pair for pair in pairs}
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"  {pair} scan error: {e}")

        n_pos = len(self.positions)
        n_wait = sum(1 for v in self.waiting_first_flip.values() if v)
        trades = self._read_trades()
        total_pnl = sum(t.get("net_pnl_pct", 0) for t in trades)
        logger.info(f"  Scan {now}: {n_pos} pos, {n_wait} waiting, {len(trades)} trades, PnL={total_pnl:+.2f}%")
        self._save_state()

    def _open_position(self, pair: str, direction: str, price: float, ind: dict):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.positions[pair] = {
            "direction": direction,
            "entry_price": price,
            "entry_time": now_str,
            "H_mfi": round(ind["H_mfi"], 1),
            "H_rsi": round(ind["H_rsi"], 1),
            "H_atr_r": round(ind["H_atr_r"], 2),
            "L_mfi": round(ind["L_mfi"], 1),
        }
        logger.info(f"  OPEN {direction} {pair} @ {_pfmt(price)} | "
                     f"MFI_h={ind['H_mfi']:.0f} RSI_h={ind['H_rsi']:.0f} ATR_r={ind['H_atr_r']:.2f} MFI_l={ind['L_mfi']:.0f}")

    def _close_position(self, pair: str, price: float, reason: str):
        pos = self.positions.get(pair)
        if not pos:
            return
        ep = pos["entry_price"]
        d = 1 if pos["direction"] == "LONG" else -1
        pnl = (price - ep) / ep * 100 * d
        fee = 0.04
        net = pnl - fee

        logger.info(f"  CLOSE {pos['direction']} {pair} @ {_pfmt(price)} | PnL {pnl:+.2f}% net {net:+.2f}% | {reason}")

        trades = self._read_trades()
        trades.append({
            "pair": pair,
            "direction": pos["direction"],
            "entry_price": ep,
            "exit_price": price,
            "pnl_pct": round(pnl, 4),
            "net_pnl_pct": round(net, 4),
            "exit_reason": reason,
            "open_time": pos["entry_time"],
            "close_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "H_mfi": pos.get("H_mfi"),
            "H_rsi": pos.get("H_rsi"),
            "H_atr_r": pos.get("H_atr_r"),
            "L_mfi": pos.get("L_mfi"),
        })
        self._write_trades(trades)
        del self.positions[pair]

    @staticmethod
    def _archive_logs():
        archive_dir = "logs_archive"
        os.makedirs(archive_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for fname in ("swing_bot.log", "swing_trades.json"):
            if os.path.exists(fname) and os.path.getsize(fname) > 0:
                shutil.copy2(fname, os.path.join(archive_dir, f"{stamp}_{fname}"))
                with open(fname, "w"):
                    pass

    def run(self, once: bool = False):
        self._archive_logs()
        logger.info(f"Swing Bot v{VERSION}")
        logger.info(f"Mode: {'PAPER' if self.paper else 'LIVE'}")
        logger.info(f"Strategy: 4h+15m Supertrend + H_MFI + H_RSI + H_ATR<1.5 + L_MFI")
        logger.info(f"Startup: WAIT for first HTF flip before entering (no blind entry)")
        logger.info(f"Pairs: {len(self.pairs)}")

        self.initialize()

        if once:
            self.scan()
            return

        logger.info("Running — synced to 15m candle close (:00, :15, :30, :45)...")
        while True:
            try:
                self._wait_for_candle_close()
                self.scan()
            except KeyboardInterrupt:
                logger.info("Stopping...")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"Scan error: {e}", exc_info=True)
                self._save_state()
                time.sleep(60)

    @staticmethod
    def _wait_for_candle_close():
        """Sleep until next 15-minute candle close + 5s buffer."""
        now = datetime.now()
        minute = now.minute
        next_quarter = ((minute // 15) + 1) * 15
        if next_quarter >= 60:
            wait_min = 60 - minute
        else:
            wait_min = next_quarter - minute
        wait_sec = wait_min * 60 - now.second + 5  # +5s buffer for candle to finalize
        if wait_sec <= 0:
            wait_sec = 1
        logger.info(f"  Waiting {wait_sec}s until next 15m close...")
        time.sleep(wait_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swing Bot")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    parser.add_argument("--live", action="store_true", help="Real trading (default: paper)")
    args = parser.parse_args()

    bot = SwingBot(paper=not args.live)
    bot.run(once=args.once)
