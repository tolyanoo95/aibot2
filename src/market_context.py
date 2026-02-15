"""
Market context: Open Interest, Long/Short ratio, stablecoin dominance,
total market cap, and liquidation zone estimation.

Data sources:
  - Binance Futures public API  (OI, L/S ratio) — no auth needed
  - CoinGecko free API           (USDT.D, USDC.D, stablecoin dom, total mcap)
"""

import time
import logging
from typing import Dict

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"
COINGECKO_API = "https://api.coingecko.com/api/v3"


def _to_binance(symbol: str) -> str:
    """'BTC/USDT' → 'BTCUSDT'."""
    return symbol.replace("/", "").replace(":", "")


class MarketContext:
    """Macro / sentiment data for ML features and LLM context."""

    def __init__(self, config):
        self.config = config
        self._sess = requests.Session()
        self._sess.headers.update({"Accept": "application/json"})
        self._global_cache: Dict = {}
        self._global_ts: float = 0
        self._CACHE_TTL = 300  # seconds

    # ══════════════════════════════════════════════════════════
    #  Binance: Open Interest
    # ══════════════════════════════════════════════════════════

    def fetch_open_interest(self, symbol: str) -> float:
        """Current OI in contracts."""
        try:
            r = self._sess.get(
                f"{BINANCE_FAPI}/fapi/v1/openInterest",
                params={"symbol": _to_binance(symbol)},
                timeout=10,
            )
            r.raise_for_status()
            return float(r.json().get("openInterest", 0))
        except Exception as exc:
            logger.debug("OI fetch %s: %s", symbol, exc)
            return 0.0

    def fetch_oi_history(
        self, symbol: str, period: str = "1h", limit: int = 500,
    ) -> pd.DataFrame:
        """Historical OI via /futures/data/openInterestHist (public)."""
        try:
            r = self._sess.get(
                f"{BINANCE_FAPI}/futures/data/openInterestHist",
                params={
                    "symbol": _to_binance(symbol),
                    "period": period,
                    "limit": limit,
                },
                timeout=15,
            )
            r.raise_for_status()
            rows = r.json()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["oi"] = df["sumOpenInterest"].astype(float)
            df["oi_value"] = df["sumOpenInterestValue"].astype(float)
            return df[["oi", "oi_value"]]
        except Exception as exc:
            logger.debug("OI history %s: %s", symbol, exc)
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════
    #  Binance: Long / Short ratio (top traders, accounts)
    # ══════════════════════════════════════════════════════════

    def fetch_long_short_ratio(
        self, symbol: str, period: str = "5m", limit: int = 1,
    ) -> Dict:
        """Latest top-trader account long/short ratio."""
        try:
            r = self._sess.get(
                f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio",
                params={
                    "symbol": _to_binance(symbol),
                    "period": period,
                    "limit": limit,
                },
                timeout=10,
            )
            r.raise_for_status()
            rows = r.json()
            if rows:
                d = rows[-1]
                return {
                    "long_short_ratio": float(d["longShortRatio"]),
                    "long_account_pct": float(d["longAccount"]),
                    "short_account_pct": float(d["shortAccount"]),
                }
        except Exception as exc:
            logger.debug("L/S %s: %s", symbol, exc)
        return {
            "long_short_ratio": 1.0,
            "long_account_pct": 0.5,
            "short_account_pct": 0.5,
        }

    def fetch_ls_history(
        self, symbol: str, period: str = "1h", limit: int = 500,
    ) -> pd.DataFrame:
        """Historical top-trader L/S ratio."""
        try:
            r = self._sess.get(
                f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio",
                params={
                    "symbol": _to_binance(symbol),
                    "period": period,
                    "limit": limit,
                },
                timeout=15,
            )
            r.raise_for_status()
            rows = r.json()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["long_short_ratio"] = df["longShortRatio"].astype(float)
            df["long_pct"] = df["longAccount"].astype(float)
            return df[["long_short_ratio", "long_pct"]]
        except Exception as exc:
            logger.debug("L/S history %s: %s", symbol, exc)
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════
    #  CoinGecko: global dominance & market cap
    # ══════════════════════════════════════════════════════════

    _STABLECOINS = [
        "usdt", "usdc", "dai", "busd", "tusd",
        "usdp", "frax", "usdd", "fdusd", "pyusd",
    ]

    def fetch_global_metrics(self) -> Dict:
        """Total market cap, BTC dom, stablecoin dominance (cached)."""
        now = time.time()
        if now - self._global_ts < self._CACHE_TTL and self._global_cache:
            return self._global_cache

        try:
            r = self._sess.get(f"{COINGECKO_API}/global", timeout=10)
            r.raise_for_status()
            d = r.json().get("data", {})

            mcap_pct = d.get("market_cap_percentage", {})
            usdt_dom = mcap_pct.get("usdt", 0)
            usdc_dom = mcap_pct.get("usdc", 0)
            stable_dom = sum(mcap_pct.get(s, 0) for s in self._STABLECOINS)

            result = {
                "total_mcap_b": d.get("total_market_cap", {}).get("usd", 0) / 1e9,
                "total_vol_b": d.get("total_volume", {}).get("usd", 0) / 1e9,
                "mcap_change_24h": d.get(
                    "market_cap_change_percentage_24h_usd", 0,
                ),
                "btc_dom": mcap_pct.get("btc", 0),
                "eth_dom": mcap_pct.get("eth", 0),
                "usdt_dom": usdt_dom,
                "usdc_dom": usdc_dom,
                "stablecoin_dom": stable_dom,
            }
            self._global_cache = result
            self._global_ts = now
            return result

        except Exception as exc:
            logger.warning("CoinGecko: %s", exc)
            return self._global_cache or self._empty_global()

    @staticmethod
    def _empty_global() -> Dict:
        return {
            "total_mcap_b": 0, "total_vol_b": 0, "mcap_change_24h": 0,
            "btc_dom": 0, "eth_dom": 0,
            "usdt_dom": 0, "usdc_dom": 0, "stablecoin_dom": 0,
        }

    # ══════════════════════════════════════════════════════════
    #  Liquidation zone estimation
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def estimate_liquidation_zones(
        df: pd.DataFrame, current_price: float, atr: float,
    ) -> Dict:
        """
        Approximate nearest liquidation clusters from swing points.

        - Swing highs above price → short-liquidation zone (squeeze risk)
        - Swing lows  below price → long-liquidation  zone (cascade risk)
        """
        default = {
            "long_liq_zone": 0.0,
            "short_liq_zone": 0.0,
            "liq_pressure": "NEUTRAL",
            "dist_to_short_liq_pct": 0.0,
            "dist_to_long_liq_pct": 0.0,
        }
        if df.empty or len(df) < 30 or atr == 0 or current_price == 0:
            return default

        lookback = min(100, len(df))
        high = df["high"].values[-lookback:]
        low = df["low"].values[-lookback:]

        swing_hi: list[float] = []
        swing_lo: list[float] = []
        WIN = 5
        for i in range(WIN, lookback - WIN):
            if high[i] == high[max(0, i - WIN): i + WIN + 1].max():
                swing_hi.append(float(high[i]))
            if low[i] == low[max(0, i - WIN): i + WIN + 1].min():
                swing_lo.append(float(low[i]))

        above = [h for h in swing_hi if h > current_price]
        below = [l for l in swing_lo if l < current_price]

        short_liq = min(above) if above else current_price + atr * 3
        long_liq = max(below) if below else current_price - atr * 3

        dist_up = (short_liq - current_price) / current_price * 100
        dist_down = (current_price - long_liq) / current_price * 100

        if dist_up < dist_down * 0.7:
            pressure = "SHORT_SQUEEZE"
        elif dist_down < dist_up * 0.7:
            pressure = "LONG_CASCADE"
        else:
            pressure = "NEUTRAL"

        return {
            "long_liq_zone": round(long_liq, 6),
            "short_liq_zone": round(short_liq, 6),
            "liq_pressure": pressure,
            "dist_to_short_liq_pct": round(dist_up, 3),
            "dist_to_long_liq_pct": round(dist_down, 3),
        }

    # ══════════════════════════════════════════════════════════
    #  Order Book Imbalance
    # ══════════════════════════════════════════════════════════

    def fetch_orderbook_imbalance(
        self, symbol: str, depth: int = 20,
    ) -> Dict:
        """
        Fetch order book and compute bid/ask imbalance.

        Returns:
          bid_ask_imbalance: -1.0 to +1.0
            +1 = all bids (strong buy pressure)
            -1 = all asks (strong sell pressure)
             0 = balanced
          bid_volume: total bid volume in top `depth` levels
          ask_volume: total ask volume in top `depth` levels
          bid_wall / ask_wall: largest single order level
        """
        try:
            r = self._sess.get(
                f"{BINANCE_FAPI}/fapi/v1/depth",
                params={"symbol": _to_binance(symbol), "limit": depth},
                timeout=5,
            )
            r.raise_for_status()
            book = r.json()

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            bid_vol = sum(float(b[1]) for b in bids)
            ask_vol = sum(float(a[1]) for a in asks)
            total = bid_vol + ask_vol

            imbalance = (bid_vol - ask_vol) / total if total > 0 else 0

            # find walls (largest single level)
            bid_wall = max((float(b[1]), float(b[0])) for b in bids) if bids else (0, 0)
            ask_wall = max((float(a[1]), float(a[0])) for a in asks) if asks else (0, 0)

            return {
                "bid_ask_imbalance": round(imbalance, 4),
                "bid_volume": round(bid_vol, 2),
                "ask_volume": round(ask_vol, 2),
                "bid_wall_price": bid_wall[1],
                "bid_wall_size": bid_wall[0],
                "ask_wall_price": ask_wall[1],
                "ask_wall_size": ask_wall[0],
            }
        except Exception as exc:
            logger.debug("Orderbook %s: %s", symbol, exc)
            return {
                "bid_ask_imbalance": 0,
                "bid_volume": 0, "ask_volume": 0,
                "bid_wall_price": 0, "bid_wall_size": 0,
                "ask_wall_price": 0, "ask_wall_size": 0,
            }

    # ══════════════════════════════════════════════════════════
    #  Aggregated context for scanner
    # ══════════════════════════════════════════════════════════

    def get_symbol_context(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Full context dict for one symbol (scanner / LLM)."""
        price = float(df["close"].iloc[-1]) if not df.empty else 0
        atr = (
            float(df["atr"].iloc[-1])
            if not df.empty and "atr" in df.columns and pd.notna(df["atr"].iloc[-1])
            else 0
        )

        oi = self.fetch_open_interest(symbol)
        ls = self.fetch_long_short_ratio(symbol)
        gm = self.fetch_global_metrics()
        liq = self.estimate_liquidation_zones(df, price, atr)
        ob = self.fetch_orderbook_imbalance(symbol)

        return {"open_interest": oi, **ls, **gm, **liq, **ob}

    # ══════════════════════════════════════════════════════════
    #  Historical context features for training
    # ══════════════════════════════════════════════════════════

    def fetch_training_context(self, symbol: str) -> pd.DataFrame:
        """
        Historical OI + L/S at 1 h resolution (public, no auth).
        Columns returned: oi_change_pct, long_short_ratio.
        Reindex / ffill to 5 m in the caller.
        """
        oi = self.fetch_oi_history(symbol, period="1h", limit=500)
        ls = self.fetch_ls_history(symbol, period="1h", limit=500)

        parts = []
        if not oi.empty:
            oi["oi_change_pct"] = oi["oi"].pct_change() * 100
            parts.append(oi[["oi_change_pct"]])
        if not ls.empty:
            parts.append(ls[["long_short_ratio"]])

        if not parts:
            return pd.DataFrame()

        merged = pd.concat(parts, axis=1, join="outer")
        return merged.ffill().fillna(0)
