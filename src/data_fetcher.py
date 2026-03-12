"""
Binance Futures data fetcher.
Supports OHLCV, funding rates, tickers, and paginated historical fetching.
"""

import time
import logging
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """Fetches market data from Binance USDⓈ-M Futures via ccxt."""

    def __init__(self, config):
        self.config = config
        self.exchange = ccxt.binanceusdm({
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if config.BINANCE_TESTNET:
            self.exchange.set_sandbox_mode(True)

    # ── public helpers ───────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles with taker buy volume and num trades from raw Binance API."""
        try:
            params = {"symbol": symbol.replace("/", "").replace(":USDT", ""), "interval": timeframe, "limit": limit}
            if since:
                params["startTime"] = since
            raw = self.exchange.fapiPublicGetKlines(params)
            rows = []
            for k in raw:
                rows.append([
                    int(k[0]),       # open time
                    float(k[1]),     # open
                    float(k[2]),     # high
                    float(k[3]),     # low
                    float(k[4]),     # close
                    float(k[5]),     # volume
                    float(k[7]),     # quote_volume
                    int(k[8]),       # num_trades
                    float(k[9]),     # taker_buy_vol
                    float(k[10]),    # taker_buy_quote_vol
                ])
            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "quote_volume", "num_trades", "taker_buy_vol", "taker_buy_quote_vol",
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as exc:
            logger.error("fetch_ohlcv %s %s: %s", symbol, timeframe, exc)
            return pd.DataFrame()

    def fetch_ohlcv_extended(
        self,
        symbol: str,
        timeframe: str = "5m",
        total_candles: int = 1500,
    ) -> pd.DataFrame:
        """Paginated fetch to get more historical data than the 1 000 limit."""
        tf_ms = self._timeframe_to_ms(timeframe)
        all_frames: list[pd.DataFrame] = []
        now_ms = int(time.time() * 1000)
        since = now_ms - total_candles * tf_ms

        while since < now_ms:
            batch = self.fetch_ohlcv(symbol, timeframe, limit=1000, since=since)
            if batch.empty:
                break
            all_frames.append(batch)
            # advance `since` past the last fetched candle
            last_ts = int(batch.index[-1].timestamp() * 1000)
            since = last_ts + tf_ms
            time.sleep(self.exchange.rateLimit / 1000)  # respect rate-limit

        if not all_frames:
            return pd.DataFrame()
        combined = pd.concat(all_frames)
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def fetch_funding_rate(self, symbol: str) -> float:
        try:
            info = self.exchange.fetch_funding_rate(symbol)
            return float(info.get("fundingRate", 0.0))
        except Exception as exc:
            logger.error("fetch_funding_rate %s: %s", symbol, exc)
            return 0.0

    def fetch_ticker(self, symbol: str) -> dict:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as exc:
            logger.error("fetch_ticker %s: %s", symbol, exc)
            return {}

    def fetch_all_data(self, symbol: str) -> dict:
        """Fetch all data needed for analysis of one symbol."""
        return {
            "primary": self.fetch_ohlcv(
                symbol, self.config.PRIMARY_TIMEFRAME, self.config.CANDLES_LIMIT,
            ),
            "secondary": self.fetch_ohlcv(
                symbol, self.config.SECONDARY_TIMEFRAME, self.config.CANDLES_LIMIT,
            ),
            "trend": self.fetch_ohlcv(
                symbol, self.config.TREND_TIMEFRAME, self.config.CANDLES_LIMIT,
            ),
            "funding_rate": self.fetch_funding_rate(symbol),
            "ticker": self.fetch_ticker(symbol),
        }

    # ── internal ─────────────────────────────────────────────

    @staticmethod
    def _timeframe_to_ms(tf: str) -> int:
        """Convert a ccxt timeframe string to milliseconds."""
        unit = tf[-1]
        amount = int(tf[:-1])
        multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
        return amount * multipliers.get(unit, 60_000)
