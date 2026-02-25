"""
Regime Classifier: determines current market state.
Outputs: TREND (1), REVERSAL (2), RANGE (0) with confidence.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "RANGE", 1: "TREND", 2: "REVERSAL"}


class RegimeClassifier:
    """LightGBM classifier for market regime detection."""

    def __init__(self, model_path: str = "models/regime_model.pkl"):
        self.model_path = model_path
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: Optional[list] = None
        self.is_trained: bool = False
        self._try_load()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list,
        n_splits: int = 5,
    ) -> Dict:
        """Train the regime classifier with time-series cross-validation."""
        self.feature_names = feature_names

        X_clean = X[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)

        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            class_weight="balanced",
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = self.model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            scores.append(acc)
            logger.info("  Fold %d: %.4f", fold + 1, acc)

        self.model.fit(X_clean, y)
        self.is_trained = True

        self._save()

        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        logger.info("Regime Classifier CV: %.4f (±%.4f)", mean_acc, std_acc)

        return {
            "cv_accuracy": mean_acc,
            "cv_std": std_acc,
            "scores": scores,
            "class_distribution": y.value_counts().to_dict(),
        }

    def predict(self, X: pd.DataFrame) -> Dict:
        """Predict market regime for the latest bar."""
        if not self.is_trained or self.model is None:
            return {"regime": "TREND", "confidence": 0.5, "probabilities": {}}

        X_aligned = self._align_features(X)
        X_last = X_aligned.iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)

        proba = self.model.predict_proba(X_last)[0]
        pred_cls = int(np.argmax(proba))
        confidence = float(proba[pred_cls])

        regime = REGIME_LABELS.get(pred_cls, "TREND")

        return {
            "regime": regime,
            "confidence": confidence,
            "probabilities": {
                "RANGE": float(proba[0]) if len(proba) > 0 else 0,
                "TREND": float(proba[1]) if len(proba) > 1 else 0,
                "REVERSAL": float(proba[2]) if len(proba) > 2 else 0,
            },
        }

    def feature_importance(self, top_n: int = 12) -> list:
        """Return top N features by importance."""
        if not self.is_trained or self.model is None:
            return []
        importance = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importance))]
        pairs = sorted(zip(names, importance), key=lambda x: -x[1])
        return pairs[:top_n]

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            return X
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = 0
        return X[self.feature_names]

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, self.model_path)
        logger.info("Regime Classifier saved → %s", self.model_path)

    def _try_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.is_trained = True
            logger.info("Regime Classifier loaded from %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load Regime Classifier: %s", exc)
