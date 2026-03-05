"""
Trend Health Model: predicts whether the current trend is HEALTHY (continues)
or ENDING (about to reverse). Binary classifier that acts as a safety layer
on top of Trend/Range/Reversal models — reduces confidence when trend is ending.
"""

import os
import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class TrendHealthModel:
    """XGBoost + LightGBM + CatBoost ensemble for trend health detection."""

    def __init__(self, model_path: str = "models/trend_health_model.pkl"):
        self.model_path = model_path
        self.models: Dict[str, object] = {}
        self.feature_names: Optional[List[str]] = None
        self.n_classes: int = 2
        self.is_trained: bool = False
        self._try_load()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list,
        n_splits: int = 5,
    ) -> Dict:
        self.feature_names = feature_names
        self.n_classes = 2

        X_clean = X[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)

        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", verbosity=0,
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1, class_weight="balanced",
        )

        cat_model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=1.0, random_seed=42, verbose=0,
            loss_function="Logloss",
        )

        model_configs = {"xgboost": xgb_model, "lightgbm": lgb_model, "catboost": cat_model}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {name: [] for name in model_configs}
        scores["ensemble"] = []

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
            cat_model.fit(X_train, y_train, sample_weight=weights,
                          eval_set=(X_val, y_val), verbose=0)

            all_proba = []
            for name, m in model_configs.items():
                proba = m.predict_proba(X_val)
                scores[name].append(accuracy_score(y_val, np.argmax(proba, axis=1)))
                all_proba.append(proba)

            ens_proba = np.mean(all_proba, axis=0)
            scores["ensemble"].append(accuracy_score(y_val, np.argmax(ens_proba, axis=1)))

        weights_full = FeatureEngineer.compute_sample_weights(y)
        xgb_model.fit(X_clean, y, sample_weight=weights_full)
        lgb_model.fit(X_clean, y)
        cat_model.fit(X_clean, y, sample_weight=weights_full, verbose=0)

        self.models = {"xgboost": xgb_model, "lightgbm": lgb_model, "catboost": cat_model}
        self.is_trained = True
        self._save()

        return {
            "cv_accuracy": np.mean(scores["ensemble"]),
            "cv_std": np.std(scores["ensemble"]),
            "scores": scores,
        }

    def predict(self, X: pd.DataFrame) -> Dict:
        """Predict trend health: HEALTHY (1) or ENDING (0)."""
        empty = {"signal": 1, "confidence": 0.5, "is_ending": False, "disagreement": 0.0}
        if not self.is_trained or not self.models:
            return empty

        X_aligned = self._align_features(X)
        X_last = X_aligned.iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)

        all_proba = []
        per_model_cls = []
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_last)[0]
                all_proba.append(proba)
                per_model_cls.append(int(np.argmax(proba)))
            except Exception as exc:
                logger.debug("TrendHealth predict error %s: %s", name, exc)

        if not all_proba:
            return empty

        avg_proba = np.mean(all_proba, axis=0)
        pred_cls = int(np.argmax(avg_proba))
        confidence = float(avg_proba[pred_cls])

        disagreement = 0.0
        if len(per_model_cls) >= 2:
            majority = max(set(per_model_cls), key=per_model_cls.count)
            n_disagree = sum(1 for c in per_model_cls if c != majority)
            disagreement = n_disagree / len(per_model_cls)

        # class 0 = ENDING, class 1 = HEALTHY
        is_ending = pred_cls == 0
        ending_conf = float(avg_proba[0])

        return {
            "signal": pred_cls,
            "confidence": confidence,
            "is_ending": is_ending,
            "ending_confidence": ending_conf,
            "disagreement": disagreement,
        }

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            return X
        X = X.copy()
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
        logger.info("Trend Health Model saved -> %s", self.model_path)

    def _try_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            data = joblib.load(self.model_path)
            self.models = data["models"]
            self.feature_names = data["feature_names"]
            self.n_classes = data.get("n_classes", 2)
            self.is_trained = True
            logger.info("Trend Health Model loaded from %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load Trend Health Model: %s", exc)
