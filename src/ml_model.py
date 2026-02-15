"""
XGBoost classifier for intraday signal prediction.
Supports training with time-series cross-validation, saving / loading, and inference.
"""

import os
import logging
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# label mapping: original → xgb class
_LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
_CLASS_TO_LABEL = {v: k for k, v in _LABEL_TO_CLASS.items()}


class MLSignalModel:
    """XGBoost-based directional classifier (SELL / HOLD / BUY)."""

    def __init__(self, model_path: str = "models/signal_model.json"):
        self.model_path = model_path
        self._meta_path = model_path.replace(".json", "_meta.pkl")
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained: bool = False
        self._try_load()

    # ── training ─────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train with 5-fold time-series CV and save the model."""
        self.feature_names = feature_names or list(X.columns)

        # clean data
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[mask].copy()
        y_mapped = y.loc[mask].map(_LABEL_TO_CLASS)
        sw = sample_weights[mask.values] if sample_weights is not None else None

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=30,
        )

        # time-series CV
        tscv = TimeSeriesSplit(n_splits=5)
        scores: list[float] = []

        for train_idx, val_idx in tscv.split(X_clean):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_tr, y_val = y_mapped.iloc[train_idx], y_mapped.iloc[val_idx]
            sw_tr = sw[train_idx] if sw is not None else None
            self.model.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            scores.append(accuracy_score(y_val, self.model.predict(X_val)))

        # final fit on full data (without early stopping)
        self.model.set_params(early_stopping_rounds=None)
        self.model.fit(X_clean, y_mapped, sample_weight=sw, verbose=False)
        self.is_trained = True
        self._save()

        avg = float(np.mean(scores))
        std = float(np.std(scores))
        logger.info("Model trained — CV accuracy %.4f (±%.4f)", avg, std)
        return {"cv_accuracy": avg, "cv_std": std, "scores": scores}

    # ── inference ─────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> Dict:
        """Return signal, confidence, and class probabilities for the last row."""
        empty = {"signal": 0, "confidence": 0.0, "probabilities": {}}
        if not self.is_trained or self.model is None or X.empty:
            return empty

        # Align features: use only columns the model was trained with,
        # add missing ones as 0, drop extra ones.
        X_aligned = self._align_features(X)
        X_last = X_aligned.iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)

        proba = self.model.predict_proba(X_last)[0]
        pred_cls = int(np.argmax(proba))
        confidence = float(proba[pred_cls])
        signal = _CLASS_TO_LABEL[pred_cls]

        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": {
                "sell": float(proba[0]),
                "hold": float(proba[1]),
                "buy": float(proba[2]),
            },
        }

    # ── feature alignment ─────────────────────────────────────

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align incoming features to the set the model was trained with.
        - Extra columns in X are dropped.
        - Missing columns are added as 0.
        """
        if not self.feature_names:
            return X

        trained = self.feature_names
        current = list(X.columns)

        if current == trained:
            return X

        extra = set(current) - set(trained)
        missing = set(trained) - set(current)

        if extra:
            logger.debug("Dropping extra features: %s", extra)
        if missing:
            logger.debug("Adding missing features (zeros): %s", missing)

        out = X.drop(columns=[c for c in extra if c in X.columns], errors="ignore")
        for col in missing:
            out[col] = 0.0

        return out[trained]  # enforce exact column order

    # ── feature importance ───────────────────────────────────

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or self.model is None:
            return {}
        imp = self.model.feature_importances_
        names = self.feature_names or [str(i) for i in range(len(imp))]
        return dict(sorted(zip(names, imp), key=lambda x: x[1], reverse=True))

    # ── persistence ──────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        self.model.save_model(self.model_path)
        joblib.dump({"feature_names": self.feature_names}, self._meta_path)
        logger.info("Model saved → %s", self.model_path)

    def _try_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            if os.path.exists(self._meta_path):
                meta = joblib.load(self._meta_path)
                self.feature_names = meta.get("feature_names")
            self.is_trained = True
            logger.info("ML model loaded from %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load model: %s", exc)
            self.model = None
            self.is_trained = False
