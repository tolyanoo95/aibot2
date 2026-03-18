# AI Crypto Futures Trading Bot — Повний контекст для аналізу

## Що це
Автоматичний бот для торгівлі крипто ф'ючерсами на Binance. Paper trading (тестовий режим). $1,000 капітал × 5x leverage = $5,000 buying power, 11 пар (BTC, ETH, BNB, SOL, ADA, DOT, LINK, DOGE, AVAX, LTC, XRP).

## Архітектура
- **Volume Bars**: замість часових свічок (1m, 15m), бот будує бари по об'єму торгів. Кожен бар = однаковий об'єм. Це дає кращу резолюцію в активний ринок і менше шуму у flat.
- **WebSocket aggTrade**: реальний час (~50ms). Кожна угода на Binance обробляється для: побудови volume bars, перевірки SL/TP, Move SL.
- **HTF Supertrend(9, 3.0)**: Higher Timeframe фільтр — volume bars з 5x порогом + Supertrend індикатор. Блокує трейди проти глобального тренду. Єдине покращення яке працює (+11.2% NET).
- **15m scan**: backup — оновлення ATR, статус позицій.

## Стратегія (rules-based, без ML)
```
1. aggTrade → accumulate volume → volume bar completion
2. ROC_12 > +0.30% → LONG signal, ROC_12 < -0.10% → SHORT signal
3. Guards: ADX > 20, RSI slope aligned, ATR expansion < 1.5
4. HTF Supertrend(9, 3.0): LONG тільки якщо HTF = UP, SHORT тільки якщо HTF = DN
5. SL = entry ± 2.0 × ATR, TP = entry ± 4.0 × ATR
6. DCA: до 3 входів, крок 1.0 ATR, ADX-dynamic max entries
7. Flip: протилежний сигнал → закрити старий, відкрити новий
8. Vol Drop: vol_avg3 < 50% of vol_ma20 → закрити (після 3 барів)
9. Timeout: max_hold = 24 volume bars
10. Cooldown: 3 бари після кожного трейду, 8 барів після 2 SL підряд
11. Move SL: 4-step [(0.25→0.30), (0.5→0.25), (1.0→0.5), (2.0→1.0)] + trail 1.0 ATR
12. Commission: maker 0.02% + taker 0.05% = 0.07% per trade
```

## Ключова проблема: Move SL обрізає вінерів

### Бектест (439 днів, 2025-2026, $1K × 5x):
| Config | NET/місяць | Трейди | WR |
|---|---|---|---|
| **No Move SL (SL=2.0 TP=4.0)** | **$1,436** | 20,384 | 53.7% |
| Current (4-step Move SL) | $339 | 20,384 | 83.4% |

Move SL підвищує WR (53% → 83%) але ЗМЕНШУЄ NET в 4 рази бо закриває трейди які дійшли б до TP (+4 ATR = +$7 кожен).

### Бектест (18 днів, Mar 1-18, 2026):
| Config | NET | Trades | TP | SL | TIMEOUT |
|---|---|---|---|---|---|
| **No Move SL** | **+$935** | 281 | 117 | 96 | 50 |
| Move SL 0.25 | +$532 | 422 | 75 | 27 | 24 |

### Live дані (18 годин на сервері, 18 трейдів):
| Activation | Симуляція NET |
|---|---|
| 0.25 (поточний) | +$5.28 (реальний з серверу) |
| 0.30 (симуляція з логів) | +$18.61 (2 трейди дійшли до TP замість скальпу) |
| 0.35 | +$8.31 (забагато HOLD → SL) |

### Проблема бектесту
Бектест працює по барах (OHLC), а live бот через WebSocket (50ms). Для Move SL це критично — бектест дає phantom profit (закриває по SL level, а не по ринковій ціні). Без Move SL бектест чесний (SL/TP перевіряються по high/low бару = реалістично).

## Що тестували (120+ варіантів) — ВСЕ НЕГАТИВНЕ крім HTF Supertrend

### Entry фільтри (всі мінус):
- Fear & Greed Index: -2.4%
- Funding Rate: -1.4%
- BTC cascade guard: -0.4%
- Taker Buy Ratio: -2.9%
- 14 типів MA (VWMA, KAMA, HMA, T3, RMA, ZLMA, etc.)
- LTF Supertrend guard: +1.9% (нестабільне)
- Pair/BTC ratio: -4.6% to -12%
- USDT/USDC dominance: не можна бектестити

### Momentum індикатори (всі мінус):
- TSI, CCI, Fisher Transform, Squeeze Momentum
- NVI, PVI, EFI, PVT (volume-price)
- OBV, MFI 20-80 (+$82 на 35 днях, але шум)

### Volatility фільтри для flat (всі мінус):
- ATR falling, BB squeeze, Range filter, Low volume
- Timeout cooldown, Dynamic ROC, ADX rising, Cross-pair regime, Min ATR
- RVI, Mass Index, Keltner Channel, Elder's Thermo, NATR, Ulcer Index

### Candle/Bar patterns (всі мінус):
- Doji filter, Inside Bar, Body ratio, Body direction
- Heikin-Ashi volume bars (6 варіантів)

### Alternative bar types (всі мінус):
- Range bars: +10% на 18 днях (шум)
- Renko filter (7 варіантів): все мінус

### S/R рівні (всі мінус):
- Pivot Points, Swing High/Low, N-bar breakout
- Distance from high/low, Mid-range filter

### Global market regime (всі мінус):
- Market breadth (count HTF UP/DN across 11 pairs)
- BTC as leader: -$499
- Average ROC confirm
- Breadth + ROC combo
- Contrarian breadth

### HTF покращення:
- **HTF Supertrend(9, 3.0)** — ЄДИНЕ що працює (+11.2%, стабільно 3 роки) ✅
- HTF ADX, RSI slope, ATR expansion, OBV, volume confirm — все мінус
- Triple EMA: +2.5% але нестабільне по роках
- HTF VWMA9 vs EMA21: +3.4% стабільне, але менше ніж Supertrend

### Move SL варіанти (39 тестів):
- Single-step (16 варіантів): найкращий 1.0→+0.5 (+20.7%)
- Multi-step (11 варіантів): 4-step найкращий (+41.2%)
- Trail (12 варіантів): 3-step + trail 1.0 найкращий (+43.1%)
- **Але ВСЕ гірше ніж no Move SL на довгому бектесті**
- Soft Move SL (тільки підтягнути SL): все мінус
- Late Move SL (+2.0→+1.0): -$283 vs no Move SL

### Інше:
- Remove ADX guard: +14.4% (pending, не deployed)
- Remove all guards (only HTF+ROC): +71.8% (pending, не deployed)
- Global MAX_OPEN=3 + ROC rank: position size inflation, потребує перевірки
- Dead hours filter: всі години прибуткові, фільтр мінус
- Pairs trading: -$4K to -$9.7K (комісії з'їдають)

## Паралельне тестування на VPS (зараз)
| Бот | Move SL Step1 | Тип |
|---|---|---|
| volbars-bot | (0.25, 0.30) | aggTrade WebSocket |
| volbars-bot-03 | (0.30, 0.35) | aggTrade WebSocket |
| volbars-bot-old | (0.15, 0.20) | 15m polling |

## Ключові файли
- `main_volbars.py` — live бот (~1562 рядків)
- `backtest_volbars.py` — головний бектест (walk-forward)
- `backtest_wf.py` — simulate_dca_trades engine
- `fast_test.py` — швидкий SL/TP grid
- `tests/test_bot.py` — 36 unit tests

## Питання для аналізу
1. Чи є спосіб покращити стратегію який ми НЕ тестували?
2. Move SL: бектест каже прибрати (4x NET), лайв показує що він рятує у flat. Яке рішення оптимальне?
3. Flat market: бот втрачає гроші у flat (комісії + timeout/SL). 120+ фільтрів не допомогли. Є інший підхід?
4. Чи варто прибрати guards (ADX, RSI slope, ATR expansion)? Бектест каже +71.8% без них, але лайв показує що guards блокують погані трейди.
5. Який оптимальний Move SL activation (0.25, 0.30, 0.35) базуючись на trade-off: скальп прибуток vs пропущені TP?
6. Чи є фундаментально інший підхід до exit management який не обрізає вінерів але зменшує ризик?
