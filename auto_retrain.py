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
    """Read accuracy from the last history entry."""
    history = load_history()
    if history:
        return history[-1].get("cv_accuracy", 0)
    return 0.0


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

    # ── train new model ──────────────────────────────────
    try:
        from train import train_model_and_return_metrics
        metrics = train_model_and_return_metrics()
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        _restore_backup(model_path, meta_path, backup_model, backup_meta)
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

    # ── restart bot to load new model ────────────────────
    if new_accuracy >= old_accuracy:
        try:
            import subprocess
            subprocess.run(["systemctl", "restart", "aibot"], timeout=10)
            logger.info("Bot restarted with new model")
        except Exception as exc:
            logger.warning("Could not restart bot: %s (restart manually)", exc)

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
