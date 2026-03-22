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
        
        self.position = {sym: None for sym in symbols} # None, 'long', 'short'
        self.entry_price = {sym: 0 for sym in symbols}
        self.entry_time = {sym: None for sym in symbols}
        self.target_tp_price = {sym: 0 for sym in symbols}
        self.target_sl_price = {sym: 0 for sym in symbols}
        self.pending_entry = {sym: None for sym in symbols} # None or {'type': 'long', 'price': 150.0, 'ts': time}
        
        # Best strategy parameters from 5m backtest (Delay + Limit Orders)
        self.entry_delay_pct = 0.015 # Wait for 1.5% drop/pump to enter
        
        # We will use slightly different TP/SL targets based on symbol
        self.tp_pct = {"SOLUSDT": 0.025, "AVAXUSDT": 0.015}
        self.sl_pct = {"SOLUSDT": 0.040, "AVAXUSDT": 0.030}
        
        # Symbol-specific thresholds
        self.oi_thresh = {
            "SOLUSDT": 5000,
            "AVAXUSDT": 500
        }
        
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

    def place_limit_order(self, symbol, side, qty, price):
        """Places a Limit order"""
        if not self.session:
            logger.info(f"🟢 [DRY RUN] Would place LIMIT {side} order for {qty} {symbol} at {price:.2f}")
            return "dry_run_id"
            
        try:
            resp = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="GTC"
            )
            logger.info(f"🟢 API Limit Order Placed: {resp}")
            return resp.get('result', {}).get('orderId')
        except Exception as e:
            logger.error(f"🔴 Failed to place Limit order: {e}")
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
                for symbol in [s for s in self.symbols if s != "BTCUSDT"]:
                    sol_latest = self.calculate_5m_bar(symbol)
                if sol_latest:
                    curr_price = sol_latest['close_price']
                    
                    # 1. Check if we need to enter a pending limit order
                    if self.pending_entry[symbol] and not self.position[symbol]:
                        # Expire pending entry after 10 hours (120 * 5m bars)
                        if (datetime.now() - self.pending_entry[symbol]['ts']).total_seconds() > 10 * 3600:
                            logger.info(f"⏳ Pending entry expired for {symbol}.")
                            self.pending_entry[symbol] = None
                        else:
                            # Check for fill
                            if self.pending_entry[symbol]['type'] == 'long' and curr_price <= self.pending_entry[symbol]['price']:
                                self.position[symbol] = 'long'
                                self.entry_price[symbol] = self.pending_entry[symbol]['price']
                                self.entry_time[symbol] = datetime.now()
                                self.target_tp_price[symbol] = self.entry_price[symbol] * (1 + self.tp_pct[symbol])
                                self.target_sl_price[symbol] = self.entry_price[symbol] * (1 - self.sl_pct[symbol])
                                self.pending_entry[symbol] = None
                                logger.info(f"🟢 FILLED LONG LIMIT on {symbol} at {self.entry_price[symbol]:.2f}. Targets: TP {self.target_tp_price[symbol]:.2f}, SL {self.target_sl_price[symbol]:.2f}")
                                
                            elif self.pending_entry[symbol]['type'] == 'short' and curr_price >= self.pending_entry[symbol]['price']:
                                self.position[symbol] = 'short'
                                self.entry_price[symbol] = self.pending_entry[symbol]['price']
                                self.entry_time[symbol] = datetime.now()
                                self.target_tp_price[symbol] = self.entry_price[symbol] * (1 - self.tp_pct[symbol])
                                self.target_sl_price[symbol] = self.entry_price[symbol] * (1 + self.sl_pct[symbol])
                                self.pending_entry[symbol] = None
                                logger.info(f"🔴 FILLED SHORT LIMIT on {symbol} at {self.entry_price[symbol]:.2f}. Targets: TP {self.target_tp_price[symbol]:.2f}, SL {self.target_sl_price[symbol]:.2f}")

                    # 2. Check open positions continuously (Stop Loss / Take Profit can happen anytime)
                    if self.position[symbol]:
                        if self.position[symbol] == "long":
                            if curr_price >= self.target_tp_price[symbol] or curr_price <= self.target_sl_price[symbol]:
                                # Exit
                                pnl_pct = (curr_price - self.entry_price[symbol]) / self.entry_price[symbol]
                                net_pnl = pnl_pct - 0.0004 # Limit entry (Maker: 0%), Market exit (Taker: 0.04% avg on Bybit)
                                logger.info(f"💰 CLOSED LONG {symbol} at {curr_price} | Net PnL: {net_pnl*100:.2f}%")
                                
                                self.log_paper_trade({
                                    'symbol': symbol, 'type': 'long', 'entry_time': self.entry_time[symbol],
                                    'exit_time': datetime.now(), 'entry_price': self.entry_price[symbol],
                                    'exit_price': curr_price, 'net_pnl_pct': net_pnl * 100
                                })
                                self.position[symbol] = None
                                
                        elif self.position[symbol] == "short":
                            if curr_price <= self.target_tp_price[symbol] or curr_price >= self.target_sl_price[symbol]:
                                pnl_pct = (self.entry_price[symbol] - curr_price) / self.entry_price[symbol]
                                net_pnl = pnl_pct - 0.0004
                                logger.info(f"💰 CLOSED SHORT {symbol} at {curr_price} | Net PnL: {net_pnl*100:.2f}%")
                                
                                self.log_paper_trade({
                                    'symbol': symbol, 'type': 'short', 'entry_time': self.entry_time[symbol],
                                    'exit_time': datetime.now(), 'entry_price': self.entry_price[symbol],
                                    'exit_price': curr_price, 'net_pnl_pct': net_pnl * 100
                                })
                                self.position[symbol] = None
                
                # Check for Entry ONLY exactly on the 5-minute close (e.g. at second 1-2 of the new candle)
                if current_minute % 5 == 0 and current_second < 5:
                    candle_ts = int(datetime.now().replace(second=0, microsecond=0).timestamp())
                    
                    # Prevent checking the same candle twice
                    if candle_ts > self.last_candle_ts:
                        self.last_candle_ts = candle_ts
                        
                        btc_bar = self.calculate_5m_bar("BTCUSDT")
                        for symbol in [s for s in self.symbols if s != "BTCUSDT"]:
                            sol_bar = self.calculate_5m_bar(symbol)
                            if btc_bar and sol_bar:
                                # We only process logic if we have both BTC and the target coin data
                                oi_t = self.oi_thresh.get(symbol, 1000)
                                
                                logger.info(f"📊 5m Close | BTC Delta: {btc_delta:.2f} | {symbol} OI Change: {sol_bar['oi_change']:.1f} | Liq: {sol_bar['liq_buy']}/{sol_bar['liq_sell']}")
                                
                                if len(rolling_btc_deltas_5m) > 10 and not self.position[symbol]:
                                    btc_delta_thresh_long = np.percentile(rolling_btc_deltas_5m, 90)
                                    btc_delta_thresh_short = np.percentile(rolling_btc_deltas_5m, 10)
                                    
                                    # Log feature row to CSV
                                    feature_row = {
                                        'ts': datetime.now().isoformat(),
                                        'symbol': symbol,
                                        'btc_delta': btc_delta,
                                        'oi_change': sol_bar['oi_change'],
                                        'liq_buy': sol_bar['liq_buy'],
                                        'liq_sell': sol_bar['liq_sell']
                                    }
                                    
                                    file_path = "swing_live_features_5m.csv"
                                    write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
                                    try:
                                        with open(file_path, "a") as f:
                                            if write_header:
                                                f.write(",".join(feature_row.keys()) + "\n")
                                            f.write(",".join([str(v) for v in feature_row.values()]) + "\n")
                                    except Exception as e:
                                        logger.error(f"Failed to write 5m features to CSV: {e}")
                                    
                                    # LONG ENTRY
                                    if (btc_delta > btc_delta_thresh_long and 
                                        sol_bar['oi_change'] > oi_t and 
                                        sol_bar['liq_buy'] < 5000): # No massive short liquidations creating fake pump
                                        
                                        limit_price = sol_bar['close_price'] * (1 - self.entry_delay_pct)
                                        logger.info(f"🚀 SWING LONG SIGNAL on {symbol}: BTC Delta {btc_delta:.2f} > {btc_delta_thresh_long:.2f}, OI {sol_bar['oi_change']} > {oi_t}")
                                        logger.info(f"⏳ Placing LONG Limit Order at {limit_price:.2f} (waiting for {self.entry_delay_pct*100}% dip)")
                                        
                                        self.pending_entry[symbol] = {
                                            'type': 'long',
                                            'price': limit_price,
                                            'ts': datetime.now()
                                        }
                                        # self.place_limit_order(symbol, "Buy", 1.0, limit_price)
                                        
                                    # SHORT ENTRY
                                    elif (btc_delta < btc_delta_thresh_short and 
                                          sol_bar['oi_change'] > oi_t and 
                                          sol_bar['liq_sell'] < 5000):
                                          
                                        limit_price = sol_bar['close_price'] * (1 + self.entry_delay_pct)
                                        logger.info(f"🩸 SWING SHORT SIGNAL on {symbol}: BTC Delta {btc_delta:.2f} < {btc_delta_thresh_short:.2f}, OI {sol_bar['oi_change']} > {oi_t}")
                                        logger.info(f"⏳ Placing SHORT Limit Order at {limit_price:.2f} (waiting for {self.entry_delay_pct*100}% pump)")
                                        
                                        self.pending_entry[symbol] = {
                                            'type': 'short',
                                            'price': limit_price,
                                            'ts': datetime.now()
                                        }
                                        # self.place_limit_order(symbol, "Sell", 1.0, limit_price)
                                        
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
        ["BTCUSDT", "SOLUSDT", "AVAXUSDT"],
        api_key=None,
        api_secret=None,
        testnet=False # Mainnet for real data
    )
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped.")