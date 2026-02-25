"""
Range Model: specialized for sideways/ranging markets.
Mean reversion strategy — buy near support, sell near resistance.
Uses smaller TP/SL and shorter hold time than trend model.
"""

import os
import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class RangeModel:
    """XGBoost + LightGBM ensemble for range/scalp trading."""

    def __init__(self, model_path: str = "models/range_model.pkl"):
        self.model_path = model_path
        self.models: Dict[str, object] = {}
        self.feature_names: Optional[List[str]] = None
        self.n_classes: int = 3
        self.is_trained: bool = False
        self._try_load()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list,
        n_splits: int = 5,
    ) -> Dict:
        """Train range model with XGBoost + LightGBM ensemble."""
        self.feature_names = feature_names
        self.n_classes = len(y.unique())

        X_clean = X[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)

        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
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
        scores = {"xgboost": [], "lightgbm": [], "ensemble": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            from src.features import FeatureEngineer
            weights = FeatureEngineer.compute_sample_weights(y_train)

            xgb_model.fit(X_train, y_train, sample_weight=weights,
                          eval_set=[(X_val, y_val)], verbose=False)
            lgb_model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])

            xgb_proba = xgb_model.predict_proba(X_val)
            lgb_proba = lgb_model.predict_proba(X_val)
            ens_proba = (xgb_proba + lgb_proba) / 2

            scores["xgboost"].append(accuracy_score(y_val, np.argmax(xgb_proba, 1)))
            scores["lightgbm"].append(accuracy_score(y_val, np.argmax(lgb_proba, 1)))
            scores["ensemble"].append(accuracy_score(y_val, np.argmax(ens_proba, 1)))

            logger.info(
                "  Fold %d: XGB=%.4f LGB=%.4f Ens=%.4f",
                fold + 1, scores["xgboost"][-1],
                scores["lightgbm"][-1], scores["ensemble"][-1],
            )

        weights_full = FeatureEngineer.compute_sample_weights(y)
        xgb_model.fit(X_clean, y, sample_weight=weights_full)
        lgb_model.fit(X_clean, y)

        self.models = {"xgboost": xgb_model, "lightgbm": lgb_model}
        self.is_trained = True
        self._save()

        return {
            "cv_accuracy": np.mean(scores["ensemble"]),
            "cv_std": np.std(scores["ensemble"]),
            "scores": scores,
        }

    def predict(self, X: pd.DataFrame) -> Dict:
        """Predict range signal: BUY (1), SELL (-1), HOLD (0)."""
        empty = {"signal": 0, "confidence": 0.0, "probabilities": {}}
        if not self.is_trained or not self.models:
            return empty

        X_aligned = self._align_features(X)
        X_last = X_aligned.iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)

        all_proba = []
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_last)[0]
                all_proba.append(proba)
            except Exception as exc:
                logger.debug("Range predict error %s: %s", name, exc)

        if not all_proba:
            return empty

        avg_proba = np.mean(all_proba, axis=0)
        pred_cls = int(np.argmax(avg_proba))
        confidence = float(avg_proba[pred_cls])

        if self.n_classes == 2:
            signal = {0: -1, 1: 1}.get(pred_cls, 0)
            probs = {"sell": float(avg_proba[0]), "hold": 0.0, "buy": float(avg_proba[1])}
        else:
            signal = {0: -1, 1: 0, 2: 1}.get(pred_cls, 0)
            probs = {
                "sell": float(avg_proba[0]),
                "hold": float(avg_proba[1]) if len(avg_proba) > 2 else 0.0,
                "buy": float(avg_proba[-1]),
            }

        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": probs,
        }

    def feature_importance(self, top_n: int = 15) -> list:
        if not self.is_trained:
            return []
        all_imp = np.zeros(len(self.feature_names))
        for model in self.models.values():
            imp = getattr(model, "feature_importances_", np.zeros(len(self.feature_names)))
            if len(imp) == len(all_imp):
                all_imp += imp
        pairs = sorted(zip(self.feature_names, all_imp), key=lambda x: -x[1])
        return pairs[:top_n]

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            return X
        for c in self.feature_names:
            if c not in X.columns:
                X[c] = 0
        return X[self.feature_names]

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump({
            "models": self.models,
            "feature_names": self.feature_names,
            "n_classes": self.n_classes,
        }, self.model_path)
        logger.info("Range Model saved → %s", self.model_path)

    def _try_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            data = joblib.load(self.model_path)
            self.models = data["models"]
            self.feature_names = data["feature_names"]
            self.n_classes = data.get("n_classes", 3)
            self.is_trained = True
            logger.info("Range Model loaded from %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load Range Model: %s", exc)
