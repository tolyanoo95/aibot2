import asyncio
import json
import logging
import os
from pybit.unified_trading import WebSocket, HTTP
from datetime import datetime, timedelta
import numpy as np
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SwingOIBot:
    def __init__(self, symbols, api_key=None, api_secret=None, testnet=False):
        self.symbols = symbols
        self.ws = WebSocket(
            testnet=testnet,
            channel_type="linear",
        )
        
        self.session = HTTP(
            testnet=testnet, 
            api_key=api_key,
            api_secret=api_secret
        ) if api_key and api_secret else None
        
        # We only need trades, liquidations, and tickers (OI). No need for full orderbook!
        self.trades_history = {sym: [] for sym in symbols}
        self.liquidations_history = {sym: [] for sym in symbols}
        self.oi_history = {sym: [] for sym in symbols}
        
        self.running = False
        
        # State for paper trading
        self.position = None # None, 'long', 'short'
        self.entry_price = 0
        self.entry_time = None
        
        # Best strategy parameters from 5m backtest
        self.tp_pct = 0.02    # 2.0% Take Profit
        self.sl_pct = 0.008   # 0.8% Stop Loss
        self.oi_thresh = 5000 # Need 5000+ new SOL entering the market
        
        # To track candles
        self.last_candle_ts = 0

    def handle_trade_message(self, msg):
        try:
            symbol = msg.get("topic", "").split(".")[-1]
            if not symbol or symbol not in self.symbols:
                return
                
            data = msg.get("data", [])
            for trade in data:
                price = float(trade.get("p"))
                size = float(trade.get("v"))
                side = trade.get("S") # Buy or Sell
                ts = int(trade.get("T"))
                
                self.trades_history[symbol].append({
                    'price': price,
                    'size': size,
                    'side': side,
                    'ts': ts
                })
                
            # Keep history for the last 6 minutes (to calculate full 5m bar)
            cutoff = int((datetime.now() - timedelta(minutes=6)).timestamp() * 1000)
            self.trades_history[symbol] = [t for t in self.trades_history[symbol] if t['ts'] > cutoff]
                
        except Exception as e:
            logger.error(f"Error handling trade msg: {e}")

    def handle_liquidation_message(self, msg):
        try:
            data = msg.get("data", [])
            if isinstance(data, dict):
                data = [data]
                
            for item in data:
                symbol = item.get("symbol")
                if not symbol or symbol not in self.symbols:
                    continue
                    
                price = float(item.get("price", 0))
                size = float(item.get("size", 0))
                side = item.get("side") # Buy or Sell
                ts = int(item.get("updatedTime", int(datetime.now().timestamp()*1000)))
                
                self.liquidations_history[symbol].append({
                    'price': price,
                    'size': size,
                    'side': side,
                    'ts': ts
                })
                
            cutoff = int((datetime.now() - timedelta(minutes=6)).timestamp() * 1000)
            self.liquidations_history[symbol] = [t for t in self.liquidations_history[symbol] if t['ts'] > cutoff]
        except Exception as e:
            pass

    def handle_ticker_message(self, msg):
        try:
            symbol = msg.get("data", {}).get("symbol")
            if not symbol or symbol not in self.symbols:
                return
            
            data = msg.get("data", {})
            if "openInterest" in data:
                oi = float(data["openInterest"])
                ts = int(msg.get("ts", int(datetime.now().timestamp()*1000)))
                self.oi_history[symbol].append({'oi': oi, 'ts': ts})
                
            cutoff = int((datetime.now() - timedelta(minutes=6)).timestamp() * 1000)
            self.oi_history[symbol] = [t for t in self.oi_history[symbol] if t['ts'] > cutoff]
        except Exception as e:
            pass
            
    def calculate_5m_bar(self, symbol):
        """Calculates features for the last rolling 5 minutes"""
        now_ms = int(datetime.now().timestamp() * 1000)
        start_5m_ago = now_ms - (5 * 60 * 1000)
        
        # Trades
        recent_trades = [t for t in self.trades_history[symbol] if t['ts'] >= start_5m_ago]
        if not recent_trades:
            return None
            
        close_price = recent_trades[-1]['price']
        
        buy_vol = sum([t['size'] for t in recent_trades if t['side'] == 'Buy'])
        sell_vol = sum([t['size'] for t in recent_trades if t['side'] == 'Sell'])
        delta = buy_vol - sell_vol
        
        # Liquidations
        recent_liqs = [t for t in self.liquidations_history[symbol] if t['ts'] >= start_5m_ago]
        liq_buy = sum([t['size'] for t in recent_liqs if t['side'] == 'Buy']) # Short liquidations
        liq_sell = sum([t['size'] for t in recent_liqs if t['side'] == 'Sell']) # Long liquidations
        
        # Open Interest
        oi_change = 0
        if self.oi_history[symbol] and len(self.oi_history[symbol]) > 1:
            current_oi = self.oi_history[symbol][-1]['oi']
            # Find OI reading closest to 5m ago
            old_oi = current_oi
            for t in self.oi_history[symbol]:
                if t['ts'] >= start_5m_ago:
                    old_oi = t['oi']
                    break
            oi_change = current_oi - old_oi
            
        return {
            'close_price': close_price,
            'volume_delta': delta,
            'liq_buy': liq_buy,
            'liq_sell': liq_sell,
            'oi_change': oi_change
        }

    def place_market_order(self, symbol, side, qty):
        """Places a Market order (We pay Taker fee, but guarantee execution)"""
        if not self.session:
            logger.info(f"🟢 [DRY RUN] Would place MARKET {side} order for {qty} {symbol}")
            return "dry_run_id"
            
        try:
            resp = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty)
            )
            logger.info(f"🟢 API Market Order Placed: {resp}")
            return resp.get('result', {}).get('orderId')
        except Exception as e:
            logger.error(f"🔴 Failed to place Market order: {e}")
            return None

    def log_paper_trade(self, trade_data):
        file_path = "swing_paper_trades.json"
        trades = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    trades = json.load(f)
            except:
                pass
                
        for k, v in trade_data.items():
            if isinstance(v, datetime):
                trade_data[k] = v.isoformat()
                
        trades.append(trade_data)
        
        with open(file_path, "w") as f:
            json.dump(trades, f, indent=4)

    async def strategy_loop(self):
        logger.info("Starting 5m Swing strategy loop...")
        
        # For tracking high delta
        rolling_btc_deltas_5m = []
        
        while self.running:
            try:
                # We want to check exactly at the close of every 5-minute candle
                # e.g., 10:00, 10:05, 10:10
                current_minute = datetime.now().minute
                current_second = datetime.now().second
                
                # Check positions continuously (Stop Loss / Take Profit can happen anytime)
                sol_latest = self.calculate_5m_bar("SOLUSDT")
                if self.position and sol_latest:
                    curr_price = sol_latest['close_price']
                    
                    if self.position == "long":
                        pnl_pct = (curr_price - self.entry_price) / self.entry_price
                        if pnl_pct >= self.tp_pct or pnl_pct <= -self.sl_pct:
                            # Exit Market
                            self.place_market_order("SOLUSDT", "Sell", 1.0)
                            
                            # Calculate net PnL after TWO Taker fees (0.055% * 2 = 0.11%)
                            net_pnl = pnl_pct - 0.0011
                            logger.info(f"💰 CLOSED LONG at {curr_price} | Net PnL: {net_pnl*100:.2f}%")
                            
                            self.log_paper_trade({
                                'symbol': 'SOLUSDT', 'type': 'long', 'entry_time': self.entry_time,
                                'exit_time': datetime.now(), 'entry_price': self.entry_price,
                                'exit_price': curr_price, 'net_pnl_pct': net_pnl * 100
                            })
                            self.position = None
                            
                    elif self.position == "short":
                        pnl_pct = (self.entry_price - curr_price) / self.entry_price
                        if pnl_pct >= self.tp_pct or pnl_pct <= -self.sl_pct:
                            self.place_market_order("SOLUSDT", "Buy", 1.0)
                            net_pnl = pnl_pct - 0.0011
                            logger.info(f"💰 CLOSED SHORT at {curr_price} | Net PnL: {net_pnl*100:.2f}%")
                            
                            self.log_paper_trade({
                                'symbol': 'SOLUSDT', 'type': 'short', 'entry_time': self.entry_time,
                                'exit_time': datetime.now(), 'entry_price': self.entry_price,
                                'exit_price': curr_price, 'net_pnl_pct': net_pnl * 100
                            })
                            self.position = None
                
                # Check for Entry ONLY exactly on the 5-minute close (e.g. at second 1-2 of the new candle)
                if current_minute % 5 == 0 and current_second < 5:
                    candle_ts = int(datetime.now().replace(second=0, microsecond=0).timestamp())
                    
                    # Prevent checking the same candle twice
                    if candle_ts > self.last_candle_ts:
                        self.last_candle_ts = candle_ts
                        
                        btc_bar = self.calculate_5m_bar("BTCUSDT")
                        sol_bar = self.calculate_5m_bar("SOLUSDT")
                        
                        if btc_bar and sol_bar:
                            btc_delta = btc_bar['volume_delta']
                            rolling_btc_deltas_5m.append(btc_delta)
                            
                            # Keep last ~3 days of 5m bars for dynamic percentiles (864 bars)
                            if len(rolling_btc_deltas_5m) > 800:
                                rolling_btc_deltas_5m = rolling_btc_deltas_5m[-800:]
                                
                            logger.info(f"📊 5m Close | BTC Delta: {btc_delta:.2f} | SOL OI Change: {sol_bar['oi_change']:.1f} | Liq: {sol_bar['liq_buy']}/{sol_bar['liq_sell']}")
                            
                            if len(rolling_btc_deltas_5m) > 10 and not self.position:
                                btc_delta_thresh_long = np.percentile(rolling_btc_deltas_5m, 90)
                                btc_delta_thresh_short = np.percentile(rolling_btc_deltas_5m, 10)
                                
                                # LONG ENTRY
                                if (btc_delta > btc_delta_thresh_long and 
                                    sol_bar['oi_change'] > self.oi_thresh and 
                                    sol_bar['liq_sell'] < 5000): # No massive long liquidations killing the trend
                                    
                                    logger.info(f"🚀 SWING LONG SIGNAL on SOL: BTC Delta {btc_delta:.2f} > {btc_delta_thresh_long:.2f}, OI {sol_bar['oi_change']} > {self.oi_thresh}")
                                    self.place_market_order("SOLUSDT", "Buy", 1.0)
                                    self.position = "long"
                                    self.entry_price = sol_bar['close_price']
                                    self.entry_time = datetime.now()
                                    
                                # SHORT ENTRY
                                elif (btc_delta < btc_delta_thresh_short and 
                                      sol_bar['oi_change'] > self.oi_thresh and 
                                      sol_bar['liq_buy'] < 5000):
                                      
                                    logger.info(f"🩸 SWING SHORT SIGNAL on SOL: BTC Delta {btc_delta:.2f} < {btc_delta_thresh_short:.2f}, OI {sol_bar['oi_change']} > {self.oi_thresh}")
                                    self.place_market_order("SOLUSDT", "Sell", 1.0)
                                    self.position = "short"
                                    self.entry_price = sol_bar['close_price']
                                    self.entry_time = datetime.now()

                await asyncio.sleep(1) # Check continuously
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}")
                await asyncio.sleep(5)

    def start(self):
        self.running = True
        
        # Subscribe to Trades, Liquidations, and Tickers. NO ORDERBOOK!
        for symbol in self.symbols:
            self.ws.trade_stream(symbol=symbol, callback=self.handle_trade_message)
            self.ws.all_liquidation_stream(symbol=symbol, callback=self.handle_liquidation_message)
            self.ws.ticker_stream(symbol=symbol, callback=self.handle_ticker_message)
            
        logger.info("WebSockets connected. Collecting 5 minutes of data before first check...")
        
        asyncio.run(self.strategy_loop())

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.warning("Running 5m Swing Bot in DRY RUN (Paper) mode.")
        
    bot = SwingOIBot(
        ["BTCUSDT", "SOLUSDT"],
        api_key=None,
        api_secret=None,
        testnet=False # Mainnet for real data
    )
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped.")