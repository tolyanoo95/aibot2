# AI Крипто Ф'ючерсний Мульти-Модельний Торговий Бот

AI-бот для торгівлі криптовалютними ф'ючерсами на Binance. Використовує **Regime Classifier** для визначення стану ринку (Тренд/Розворот/Рендж), потім застосовує спеціалізовану модель для кожного режиму.

## Архітектура

```
Binance Futures API
     │
     ▼
┌──────────────────────────┐
│   Пайплайн даних          │
│   OHLCV (5m/15m/1h)      │
│   Funding Rate, OI, L/S  │
│   Стакан ордерів, лікві  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Інженерія фіч           │
│   58 базових + 13 reversal│
│   + 7 range + 12 regime   │
│   = 90 фіч загалом        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Regime Classifier       │
│   ТРЕНД / РОЗВОРОТ / РЕНДЖ│
│   LightGBM (72.9%)       │
└─────┬──────┬──────┬──────┘
      │      │      │
      ▼      ▼      ▼
┌────────┐┌────────┐┌────────┐
│ Trend  ││Reversal││ Range  │
│ Модель ││ Модель ││ Модель │
│XGB+LGB ││XGB+LGB ││XGB+LGB │
│+CatB   ││        ││        │
│71.6%   ││ 90.8%  ││ 76.0%  │
└───┬────┘└───┬────┘└───┬────┘
    │         │         │
    ▼         ▼         ▼
┌──────────────────────────┐
│   Генератор сигналів      │
│   Динамічний SL/TP        │
│   Тренд: 2.5x ATR        │
│   Розворот: 1.5x ATR     │
│   Рендж: 1.0x ATR        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Виконавець угод         │
│   Розмір позиції по conf  │
│   Захист від розворотів   │
│   Моніторинг просадки     │
│   Paper / Live режим      │
└──────────────────────────┘
```

## Моделі

| Модель | Тип | Accuracy | Призначення |
|--------|-----|----------|-------------|
| **Trend Model** | XGBoost + LightGBM + CatBoost | 71.6% | Слідувати трендам (BUY/SELL) |
| **Regime Classifier** | LightGBM | 72.9% | Визначити стан ринку |
| **Reversal Model** | XGBoost + LightGBM | 90.8% | Ловити розвороти |
| **Range Model** | XGBoost + LightGBM | 76.0% | Скальпінг у боковому ринку |

## Швидкий старт

```bash
# Клонувати
git clone https://github.com/tolyanoo95/aibot2.git
cd aibot2

# Встановити залежності
pip install -r requirements.txt

# Налаштувати
cp .env.example .env
# Відредагувати .env — додати Binance API ключі

# Натренувати всі моделі (паралельно, ~7 хв)
python train_all.py

# Один скан
python main.py --once

# Безперервний сканер
python main.py
```

## Тренування

```bash
# Натренувати всі 4 моделі одразу (рекомендовано)
python train_all.py

# Або окремо
python train.py            # Trend Model
python train_regime.py     # Regime Classifier
python train_reversal.py   # Reversal Model
python train_range.py      # Range Model
```

## Конфігурація

Основні налаштування в `src/config.py` та `.env`:

| Параметр | За замовч. | Опис |
|----------|-----------|------|
| `TRADING_MODE` | paper | paper або live |
| `PREDICTION_THRESHOLD` | 0.70 | Мін. confidence для відкриття |
| `SL_ATR_MULTIPLIER` | 2.5 | Відстань стоп-лосу (динамічна по режиму) |
| `TP_ATR_MULTIPLIER` | 2.5 | Відстань тейк-профіту (динамічна по режиму) |
| `MAX_OPEN_TRADES` | 11 | Макс. одночасних позицій |
| `MAX_HOLD_BARS` | 12 | Макс. час утримання (60 хв на 5m) |
| `TRAIN_CANDLES` | 30000 | Дані для тренування (~104 дні) |

## Управління ризиками

- **3 SL підряд** → пауза 2 години на всю торгівлю
- **SL на парі** → кулдаун 40 хв на цю пару
- **Динамічний поріг** → підвищується після послідовних збитків
- **Розмір позиції по confidence** → 90%=1.5x, 80%=1x, 70%=0.5x
- **Моніторинг просадки** → зменшення позицій якщо PnL впав на 3% від піку
- **BTC як лідер** → не торгувати альти якщо BTC невизначений

## Структура файлів

```
├── main.py                 # Головний цикл сканера
├── train_all.py            # Тренування всіх моделей (паралельно)
├── train.py                # Тренування Trend Model
├── train_regime.py         # Тренування Regime Classifier
├── train_reversal.py       # Тренування Reversal Model
├── train_range.py          # Тренування Range Model
├── auto_retrain.py         # Автоматичне перенавчання (cron)
├── backtest.py             # Бектестинг
├── src/
│   ├── config.py           # Конфігурація
│   ├── data_fetcher.py     # Дані з Binance
│   ├── indicators.py       # Технічні індикатори
│   ├── features.py         # Інженерія фіч (90 фіч)
│   ├── ml_model.py         # Trend Model (ансамбль)
│   ├── regime_classifier.py # Regime Classifier
│   ├── reversal_model.py   # Reversal Model
│   ├── range_model.py      # Range Model
│   ├── signal_generator.py # Генерація сигналів
│   ├── executor.py         # Виконання угод
│   ├── trade_monitor.py    # Моніторинг угод
│   ├── entry_refiner.py    # Уточнення входу по 1m
│   ├── market_context.py   # OI, L/S, ліквідації
│   ├── llm_analyzer.py     # LLM аналіз (опціонально)
│   └── display.py          # Консольне відображення
├── models/                 # Натреновані моделі
└── .env                    # API ключі
```

## Торгові пари

BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC (USDT ф'ючерси)
