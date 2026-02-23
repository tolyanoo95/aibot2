#!/usr/bin/env python3
"""
Auto-retrain: trains a new model on fresh data and replaces the old one
only if the new model has better (or equal) cross-validation accuracy.

Usage:
    python auto_retrain.py              # run once
    crontab -e                          # schedule it:
    0 3 */2 * * cd /path/to/aibot && python auto_retrain.py >> retrain.log 2>&1

The line above runs every 2 days at 03:00 AM.
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("retrain.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

HISTORY_FILE = "models/retrain_history.json"


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list):
    os.makedirs(os.path.dirname(HISTORY_FILE) or ".", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def get_current_accuracy() -> float:
    """Read accuracy from the model meta file, falling back to history."""
    import pickle
    from src.config import config
    meta_path = config.ML_MODEL_PATH.replace(".json", "_meta.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            acc = meta.get("cv_accuracy", 0)
            if acc > 0:
                return acc
        except Exception:
            pass
    history = load_history()
    if history:
        return history[-1].get("cv_accuracy", 0)
    return 0.0


def _stop_bot():
    """Stop the bot to avoid Binance rate limits during training."""
    import subprocess
    try:
        subprocess.run(["systemctl", "stop", "aibot"], timeout=10)
        logger.info("Bot stopped before training")
    except Exception as exc:
        logger.warning("Could not stop bot: %s", exc)


def _start_bot():
    """Start the bot after training."""
    import subprocess
    try:
        subprocess.run(["systemctl", "start", "aibot"], timeout=10)
        logger.info("Bot started with new model")
    except Exception as exc:
        logger.warning("Could not start bot: %s (start manually)", exc)


def auto_retrain():
    from src.config import config

    logger.info("=" * 50)
    logger.info("Auto-retrain started at %s", datetime.now().isoformat())

    old_accuracy = get_current_accuracy()
    logger.info("Current model accuracy: %.4f", old_accuracy)

    # ── backup current model ─────────────────────────────
    model_path = config.ML_MODEL_PATH
    meta_path = model_path.replace(".json", "_meta.pkl")
    backup_model = model_path + ".bak"
    backup_meta = meta_path + ".bak"

    if os.path.exists(model_path):
        shutil.copy2(model_path, backup_model)
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, backup_meta)
    logger.info("Old model backed up")

    # ── stop bot to avoid Binance rate limits ─────────────
    _stop_bot()

    # ── train new model ──────────────────────────────────
    try:
        from train import train_model_and_return_metrics
        metrics = train_model_and_return_metrics()
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        _restore_backup(model_path, meta_path, backup_model, backup_meta)
        _start_bot()
        return

    new_accuracy = metrics.get("cv_accuracy", 0)
    logger.info("New model accuracy: %.4f", new_accuracy)

    # ── decide: keep or rollback ─────────────────────────
    if new_accuracy >= old_accuracy:
        improvement = new_accuracy - old_accuracy
        logger.info(
            "NEW model accepted (%.4f → %.4f, %+.4f)",
            old_accuracy, new_accuracy, improvement,
        )
        # clean backups
        for f in (backup_model, backup_meta):
            if os.path.exists(f):
                os.remove(f)
    else:
        logger.warning(
            "NEW model REJECTED (%.4f < %.4f) — rolling back",
            new_accuracy, old_accuracy,
        )
        _restore_backup(model_path, meta_path, backup_model, backup_meta)

    # ── save history ─────────────────────────────────────
    history = load_history()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "cv_accuracy": new_accuracy,
        "cv_std": metrics.get("cv_std", 0),
        "accepted": new_accuracy >= old_accuracy,
        "old_accuracy": old_accuracy,
        "fold_scores": metrics.get("scores", []),
    })
    # keep last 50 entries
    save_history(history[-50:])

    # ── start bot with new (or restored) model ──────────
    _start_bot()

    logger.info("Auto-retrain complete.\n")


def _restore_backup(model_path, meta_path, backup_model, backup_meta):
    """Restore backed-up model files."""
    if os.path.exists(backup_model):
        shutil.move(backup_model, model_path)
    if os.path.exists(backup_meta):
        shutil.move(backup_meta, meta_path)
    logger.info("Old model restored from backup")


if __name__ == "__main__":
    auto_retrain()
