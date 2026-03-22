import asyncio
import json
import logging
import os
from pybit.unified_trading import WebSocket
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderbookBot:
    def __init__(self, symbols, api_key=None, api_secret=None, testnet=False):
        self.symbols = symbols
        import pybit
        self.ws = WebSocket(
            testnet=testnet,
            channel_type="linear",
        )
        # Assuming you want to use paper trading/testnet later, we can initialize pybit HTTP
        from pybit.unified_trading import HTTP
        
        self.session = HTTP(
            testnet=testnet, 
            api_key=api_key,
            api_secret=api_secret
        ) if api_key and api_secret else None
        
        self.orderbooks = {sym: {'bids': {}, 'asks': {}} for sym in symbols}
        self.trades_history = {sym: [] for sym in symbols}
        self.running = False
        
        # Virtual execution tracking
        self.virtual_orders = {} # {symbol: {'side': 'Buy', 'price': 150.0, 'status': 'open', 'id': '...'}}
        
    def handle_orderbook_message(self, msg):
        try:
            symbol = msg.get("data", {}).get("s")
            if not symbol or symbol not in self.symbols:
                return
                
            msg_type = msg.get("type")
            data = msg.get("data", {})
            
            if msg_type == "snapshot":
                self.orderbooks[symbol]['bids'] = {float(price): float(size) for price, size in data.get('b', [])}
                self.orderbooks[symbol]['asks'] = {float(price): float(size) for price, size in data.get('a', [])}
                
            elif msg_type == "delta":
                # Update bids
                for price, size in data.get('b', []):
                    p = float(price)
                    s = float(size)
                    if s == 0:
                        self.orderbooks[symbol]['bids'].pop(p, None)
                    else:
                        self.orderbooks[symbol]['bids'][p] = s
                        
                # Update asks
                for price, size in data.get('a', []):
                    p = float(price)
                    s = float(size)
                    if s == 0:
                        self.orderbooks[symbol]['asks'].pop(p, None)
                    else:
                        self.orderbooks[symbol]['asks'][p] = s
                        
        except Exception as e:
            logger.error(f"Error handling orderbook msg: {e}")

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
                
            # Keep only last 1000 trades
            if len(self.trades_history[symbol]) > 1000:
                self.trades_history[symbol] = self.trades_history[symbol][-1000:]
                
        except Exception as e:
            logger.error(f"Error handling trade msg: {e}")
            
    def calculate_features(self, symbol):
        # Calculate OBI
        bids = sorted(self.orderbooks[symbol]['bids'].items(), key=lambda x: x[0], reverse=True)
        asks = sorted(self.orderbooks[symbol]['asks'].items(), key=lambda x: x[0])
        
        # We need to ensure we only look at prices close to the current market
        # Sometimes WebSocket sends deep levels or empty levels with 0 size
        # Our update logic already pops size==0, but just in case we filter out outliers
        if not bids or not asks:
            return {'obi_10': 0, 'delta_10s': 0, 'best_bid': None, 'best_ask': None}
            
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        # Filter out obvious errors (e.g. bid > ask or ridiculously far)
        if best_bid > best_ask:
            # Re-sync needed, but for now just use the first few that make sense
            pass
            
        top_bids = bids[:10]
        top_asks = asks[:10]
        
        bid_vol = sum([b[1] for b in top_bids])
        ask_vol = sum([a[1] for a in top_asks])
        
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
        
        # Calculate Delta
        now_ms = int(datetime.now().timestamp() * 1000)
        # Look back 10 seconds
        recent_trades = [t for t in self.trades_history[symbol] if now_ms - t['ts'] < 10000]
        
        buy_vol = sum([t['size'] for t in recent_trades if t['side'] == 'Buy'])
        sell_vol = sum([t['size'] for t in recent_trades if t['side'] == 'Sell'])
        delta = buy_vol - sell_vol
        
        return {
            'obi_10': obi,
            'delta_10s': delta,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'bid_vol_10': bid_vol,
            'ask_vol_10': ask_vol
        }

    def place_maker_order(self, symbol, side, price, qty):
        """Places a PostOnly Limit order to ensure Maker fee (no Taker fee)"""
        # In DRY RUN mode on MAINNET we use Virtual Orders
        order_id = f"virt_{int(datetime.now().timestamp()*1000)}"
        self.virtual_orders[symbol] = {
            'id': order_id,
            'side': side,
            'price': price,
            'qty': qty,
            'status': 'open',
            'created_at': datetime.now()
        }
        logger.info(f"🟢 [VIRTUAL] Placed Maker {side} order at {price}")
        return order_id

    def cancel_all_orders(self, symbol):
        if symbol in self.virtual_orders:
            logger.info(f"⭕️ [VIRTUAL] Cancelled order {self.virtual_orders[symbol]['id']}")
            del self.virtual_orders[symbol]

    def check_virtual_fills(self, symbol):
        """Checks if the virtual order got filled by market action"""
        if symbol not in self.virtual_orders:
            return False
            
        order = self.virtual_orders[symbol]
        if order['status'] == 'filled':
            return True
            
        # Check trades history to see if price traded through our limit
        recent_trades = [t for t in self.trades_history[symbol] if t['ts'] > int(order['created_at'].timestamp()*1000)]
        
        for t in recent_trades:
            if order['side'] == 'Buy' and t['price'] <= order['price']:
                logger.info(f"✅ [VIRTUAL] Buy Order FILLED at {order['price']} (Market traded at {t['price']})")
                order['status'] = 'filled'
                order['fill_time'] = datetime.fromtimestamp(t['ts']/1000)
                return True
            elif order['side'] == 'Sell' and t['price'] >= order['price']:
                logger.info(f"✅ [VIRTUAL] Sell Order FILLED at {order['price']} (Market traded at {t['price']})")
                order['status'] = 'filled'
                order['fill_time'] = datetime.fromtimestamp(t['ts']/1000)
                return True
                
        return False

    def log_paper_trade(self, trade_data):
        file_path = "paper_trades.json"
        trades = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    trades = json.load(f)
            except:
                pass
                
        # Convert datetime to string for JSON serialization
        for k, v in trade_data.items():
            if isinstance(v, datetime):
                trade_data[k] = v.isoformat()
                
        trades.append(trade_data)
        
        with open(file_path, "w") as f:
            json.dump(trades, f, indent=4)

    async def strategy_loop(self):
        logger.info("Starting strategy loop...")
        
        # State for paper trading
        in_position = False
        position_type = None
        entry_price = 0
        entry_time = None
        
        tp_pct = 0.006
        sl_pct = 0.003
        
        # For tracking high delta
        rolling_deltas = []
        
        while self.running:
            try:
                btc_features = self.calculate_features("BTCUSDT")
                sol_features = self.calculate_features("SOLUSDT")
                
                # Check virtual fills first
                order_filled = self.check_virtual_fills("SOLUSDT")
                
                # If we have an open order that just got filled, update position state
                if order_filled and not in_position:
                    # Entry order was filled
                    in_position = True
                    vo = self.virtual_orders["SOLUSDT"]
                    position_type = "long" if vo['side'] == 'Buy' else "short"
                    entry_price = vo['price']
                    entry_time = vo.get('fill_time', datetime.now())
                    logger.info(f"🎯 POSITION OPENED: {position_type.upper()} at {entry_price}")
                    del self.virtual_orders["SOLUSDT"] # Clean up filled order
                elif order_filled and in_position:
                    # Exit order was filled
                    vo = self.virtual_orders["SOLUSDT"]
                    exit_price = vo['price']
                    exit_time = vo.get('fill_time', datetime.now())
                    pnl_pct = (exit_price - entry_price) / entry_price if position_type == "long" else (entry_price - exit_price) / entry_price
                    hold_time = (exit_time - entry_time).total_seconds()
                    
                    logger.info(f"💰 POSITION CLOSED at {exit_price} | PnL: {pnl_pct*100:.3f}% | Held: {hold_time}s")
                    
                    # Log to file
                    self.log_paper_trade({
                        'symbol': 'SOLUSDT',
                        'type': position_type,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct * 100,
                        'hold_time_sec': hold_time
                    })
                    
                    in_position = False
                    position_type = None
                    entry_price = 0
                    del self.virtual_orders["SOLUSDT"]
                
                # Update BTC delta history
                if btc_features['delta_10s'] != 0:
                    rolling_deltas.append(btc_features['delta_10s'])
                    if len(rolling_deltas) > 1000:
                        rolling_deltas = rolling_deltas[-1000:]
                
                if len(rolling_deltas) > 100 and sol_features['best_bid'] and sol_features['best_ask']:
                    btc_delta_thresh_long = np.percentile(rolling_deltas, 95)
                    btc_delta_thresh_short = np.percentile(rolling_deltas, 5)
                    
                    sol_mid_price = (sol_features['best_bid'] + sol_features['best_ask']) / 2
                    
                    if len(rolling_deltas) % 10 == 0:
                        # Log without spamming every second
                        logger.info(f"Monitor -> BTC Delta: {btc_features['delta_10s']:.2f} | SOL OBI: {sol_features['obi_10']:.2f} | SOL Mid: {sol_mid_price}")
                        
                    # If we don't have a position AND don't have an open entry order waiting
                    if not in_position and "SOLUSDT" not in self.virtual_orders:
                        # LONG SIGNAL
                        if btc_features['delta_10s'] > btc_delta_thresh_long and sol_features['obi_10'] > 0.1:
                            logger.info(f"🚀 LONG SIGNAL on SOL: BTC Delta {btc_features['delta_10s']:.2f}, SOL OBI {sol_features['obi_10']:.2f} | Bid: {sol_features['best_bid']}")
                            self.place_maker_order("SOLUSDT", "Buy", sol_features['best_bid'], 1.0)
                            
                        # SHORT SIGNAL
                        elif btc_features['delta_10s'] < btc_delta_thresh_short and sol_features['obi_10'] < -0.1:
                            logger.info(f"🩸 SHORT SIGNAL on SOL: BTC Delta {btc_features['delta_10s']:.2f}, SOL OBI {sol_features['obi_10']:.2f} | Ask: {sol_features['best_ask']}")
                            self.place_maker_order("SOLUSDT", "Sell", sol_features['best_ask'], 1.0)
                            
                    elif in_position and "SOLUSDT" not in self.virtual_orders:
                        # We are in position but haven't placed an exit limit order yet
                        # Or we cancelled it because we needed to update price
                        if position_type == "long":
                            pnl_pct = (sol_mid_price - entry_price) / entry_price
                            
                            exit_cond = pnl_pct >= tp_pct or pnl_pct <= -sl_pct or sol_features['obi_10'] < -0.1
                                
                            if exit_cond:
                                logger.info(f"Triggering Long Exit (PnL: {pnl_pct*100:.3f}%). Placing Limit Sell at {sol_features['best_ask']}")
                                self.place_maker_order("SOLUSDT", "Sell", sol_features['best_ask'], 1.0)
                                
                        elif position_type == "short":
                            pnl_pct = (entry_price - sol_mid_price) / entry_price
                            
                            exit_cond = pnl_pct >= tp_pct or pnl_pct <= -sl_pct or sol_features['obi_10'] > 0.1
                                
                            if exit_cond:
                                logger.info(f"Triggering Short Exit (PnL: {pnl_pct*100:.3f}%). Placing Limit Buy at {sol_features['best_bid']}")
                                self.place_maker_order("SOLUSDT", "Buy", sol_features['best_bid'], 1.0)
                                
                    elif "SOLUSDT" in self.virtual_orders:
                        # We have an open order waiting to be filled.
                        # If price moved away, we might want to cancel and replace (Chasing)
                        vo = self.virtual_orders["SOLUSDT"]
                        if vo['side'] == 'Buy' and sol_features['best_bid'] > vo['price']:
                            logger.info("Price moved up, cancelling Buy limit to chase...")
                            self.cancel_all_orders("SOLUSDT")
                        elif vo['side'] == 'Sell' and sol_features['best_ask'] < vo['price']:
                            logger.info("Price moved down, cancelling Sell limit to chase...")
                            self.cancel_all_orders("SOLUSDT")

                await asyncio.sleep(1) # Run check every 1 second
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    def start(self):
        self.running = True
        
        # Subscribe to Orderbooks
        for symbol in self.symbols:
            self.ws.orderbook_stream(
                depth=50,
                symbol=symbol,
                callback=self.handle_orderbook_message
            )
            
            self.ws.trade_stream(
                symbol=symbol,
                callback=self.handle_trade_message
            )
            
        logger.info("WebSockets connected.")
        
        # Start async loop
        asyncio.run(self.strategy_loop())

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("BYBIT_TESTNET_API_KEY")
    api_secret = os.getenv("BYBIT_TESTNET_API_SECRET")
    
    if not api_key:
        logger.warning("No BYBIT_TESTNET_API_KEY found in .env, running in DRY RUN mode without real orders.")
        
    bot = OrderbookBot(
        ["BTCUSDT", "SOLUSDT"],
        api_key=api_key,
        api_secret=api_secret,
        testnet=True # Assuming you want to use testnet for real paper trading
    )
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped.")
