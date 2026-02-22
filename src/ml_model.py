"""
Ensemble ML model: XGBoost + LightGBM + CatBoost.
Each model votes independently; final prediction is averaged probability.
"""

import os
import logging
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# label mapping: original → class (auto-detected in train())
_LABEL_TO_CLASS_3 = {-1: 0, 0: 1, 1: 2}
_CLASS_TO_LABEL_3 = {v: k for k, v in _LABEL_TO_CLASS_3.items()}

_LABEL_TO_CLASS_2 = {-1: 0, 1: 1}
_CLASS_TO_LABEL_2 = {v: k for k, v in _LABEL_TO_CLASS_2.items()}

# backward compat aliases (default 3-class)
_LABEL_TO_CLASS = _LABEL_TO_CLASS_3
_CLASS_TO_LABEL = _CLASS_TO_LABEL_3


class MLSignalModel:
    """Ensemble of XGBoost + LightGBM + CatBoost for signal prediction."""

    def __init__(self, model_path: str = "models/signal_model.json"):
        self.model_path = model_path
        self._meta_path = model_path.replace(".json", "_meta.pkl")
        self.models: Dict[str, object] = {}  # name → trained model
        self.feature_names: Optional[List[str]] = None
        self.n_classes: int = 3
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
        """Train all 3 models with time-series CV and save."""
        self.feature_names = feature_names or list(X.columns)

        # clean data
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[mask].copy()

        # auto-detect binary vs 3-class
        unique_labels = set(y.loc[mask].unique())
        if unique_labels <= {-1, 1}:
            self.n_classes = 2
            label_map = _LABEL_TO_CLASS_2
            logger.info("Binary mode: BUY vs SELL (2 classes)")
        else:
            self.n_classes = 3
            label_map = _LABEL_TO_CLASS_3
            logger.info("3-class mode: BUY / SELL / HOLD")

        y_mapped = y.loc[mask].map(label_map)
        sw = sample_weights[mask.values] if sample_weights is not None else None

        nc = self.n_classes

        # ── define models ────────────────────────────────────
        xgb_params = dict(
            n_estimators=500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        lgb_params = dict(
            n_estimators=500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        cat_params = dict(
            iterations=500, depth=7, learning_rate=0.03,
            l2_leaf_reg=1.0, random_seed=42, verbose=0,
        )

        if nc == 2:
            xgb_params.update(objective="binary:logistic", eval_metric="logloss")
            lgb_params.update(objective="binary")
            cat_params.update(loss_function="Logloss")
        else:
            xgb_params.update(objective="multi:softprob", num_class=nc, eval_metric="mlogloss")
            lgb_params.update(objective="multiclass", num_class=nc)
            cat_params.update(loss_function="MultiClass", classes_count=nc)

        model_configs = {
            "xgboost": xgb.XGBClassifier(**xgb_params),
            "lightgbm": lgb.LGBMClassifier(**lgb_params),
            "catboost": CatBoostClassifier(**cat_params),
        }

        # ── time-series CV (using XGBoost for scoring) ───────
        tscv = TimeSeriesSplit(n_splits=5)
        all_scores: Dict[str, list] = {name: [] for name in model_configs}

        for train_idx, val_idx in tscv.split(X_clean):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_tr, y_val = y_mapped.iloc[train_idx], y_mapped.iloc[val_idx]
            sw_tr = sw[train_idx] if sw is not None else None

            for name, model in model_configs.items():
                if name == "catboost":
                    model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=(X_val, y_val), verbose=0)
                elif name == "lightgbm":
                    model.fit(X_tr, y_tr, sample_weight=sw_tr)
                else:
                    model.fit(X_tr, y_tr, sample_weight=sw_tr,
                              eval_set=[(X_val, y_val)], verbose=False)

                score = accuracy_score(y_val, model.predict(X_val))
                all_scores[name].append(score)

        # ── final fit on all data ────────────────────────────
        for name, model in model_configs.items():
            logger.info("Training %s on full data …", name)
            if name == "catboost":
                model.fit(X_clean, y_mapped, sample_weight=sw, verbose=0)
            elif name == "lightgbm":
                model.fit(X_clean, y_mapped, sample_weight=sw)
            else:
                model.fit(X_clean, y_mapped, sample_weight=sw, verbose=False)

        self.models = model_configs
        self.is_trained = True
        self._save()

        # ── report per-model and ensemble scores ─────────────
        ensemble_scores = []
        for i in range(5):
            fold_avg = np.mean([all_scores[name][i] for name in model_configs])
            ensemble_scores.append(fold_avg)

        for name in model_configs:
            avg = np.mean(all_scores[name])
            logger.info("  %s CV: %.4f (±%.4f)", name, avg, np.std(all_scores[name]))

        avg = float(np.mean(ensemble_scores))
        std = float(np.std(ensemble_scores))
        logger.info("Ensemble CV accuracy %.4f (±%.4f)", avg, std)

        return {
            "cv_accuracy": avg,
            "cv_std": std,
            "scores": ensemble_scores,
            "per_model": {
                name: {
                    "cv_accuracy": float(np.mean(scores)),
                    "cv_std": float(np.std(scores)),
                }
                for name, scores in all_scores.items()
            },
        }

    # ── inference ─────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> Dict:
        """Average probabilities from all models, return signal."""
        empty = {"signal": 0, "confidence": 0.0, "probabilities": {}}
        if not self.is_trained or not self.models or X.empty:
            return empty

        X_aligned = self._align_features(X)
        X_last = X_aligned.iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)

        # collect probabilities from each model
        all_proba = []
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_last)[0]
                if len(proba) == self.n_classes:
                    all_proba.append(proba)
            except Exception as exc:
                logger.debug("Predict error %s: %s", name, exc)

        if not all_proba:
            return empty

        avg_proba = np.mean(all_proba, axis=0)
        pred_cls = int(np.argmax(avg_proba))
        confidence = float(avg_proba[pred_cls])

        if self.n_classes == 2:
            cls_map = _CLASS_TO_LABEL_2
            signal = cls_map[pred_cls]
            return {
                "signal": signal,
                "confidence": confidence,
                "probabilities": {
                    "sell": float(avg_proba[0]),
                    "hold": 0.0,
                    "buy": float(avg_proba[1]),
                },
            }
        else:
            signal = _CLASS_TO_LABEL_3[pred_cls]
            return {
                "signal": signal,
                "confidence": confidence,
                "probabilities": {
                    "sell": float(avg_proba[0]),
                    "hold": float(avg_proba[1]),
                    "buy": float(avg_proba[2]),
                },
            }

    # ── feature alignment ─────────────────────────────────────

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return X
        trained = self.feature_names
        current = list(X.columns)
        if current == trained:
            return X
        extra = set(current) - set(trained)
        missing = set(trained) - set(current)
        out = X.drop(columns=[c for c in extra if c in X.columns], errors="ignore")
        for col in missing:
            out[col] = 0.0
        return out[trained]

    # ── feature importance (averaged) ────────────────────────

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or not self.models:
            return {}

        all_imp: Dict[str, list] = {}
        names = self.feature_names or []

        for model_name, model in self.models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    for i, name in enumerate(names):
                        if name not in all_imp:
                            all_imp[name] = []
                        if i < len(imp):
                            all_imp[name].append(imp[i])
            except Exception:
                pass

        averaged = {name: float(np.mean(vals)) for name, vals in all_imp.items()}
        return dict(sorted(averaged.items(), key=lambda x: x[1], reverse=True))

    # ── persistence ──────────────────────────────────────────

    def _save(self):
        base_dir = os.path.dirname(self.model_path) or "."
        os.makedirs(base_dir, exist_ok=True)

        # save each model
        paths = {}
        for name, model in self.models.items():
            if name == "xgboost":
                path = os.path.join(base_dir, "ensemble_xgboost.json")
                model.save_model(path)
            elif name == "lightgbm":
                path = os.path.join(base_dir, "ensemble_lightgbm.pkl")
                joblib.dump(model, path)
            elif name == "catboost":
                path = os.path.join(base_dir, "ensemble_catboost.cbm")
                model.save_model(path)
            paths[name] = path

        # save metadata
        joblib.dump({
            "feature_names": self.feature_names,
            "model_paths": paths,
            "model_names": list(self.models.keys()),
            "n_classes": self.n_classes,
        }, self._meta_path)

        # keep backward compatibility — save XGBoost as main model too
        if "xgboost" in self.models:
            self.models["xgboost"].save_model(self.model_path)

        logger.info("Ensemble saved → %s", base_dir)

    def _try_load(self):
        # try loading meta
        meta = {}
        if os.path.exists(self._meta_path):
            try:
                meta = joblib.load(self._meta_path)
                self.feature_names = meta.get("feature_names")
                self.n_classes = meta.get("n_classes", 3)
            except Exception:
                pass

        # detect format: new ensemble has "model_names", old has only "feature_names"
        is_ensemble_format = "model_names" in meta

        if not is_ensemble_format:
            # legacy: single XGBoost model
            if os.path.exists(self.model_path):
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(self.model_path)
                    self.models = {"xgboost": model}
                    self.is_trained = True
                    logger.info("Legacy XGBoost model loaded from %s", self.model_path)
                except Exception as exc:
                    logger.warning("Could not load legacy model: %s", exc)
            return

        try:
            paths = meta.get("model_paths", {})
            names = meta.get("model_names", [])

            base_dir = os.path.dirname(self.model_path) or "."

            for name in names:
                path = paths.get(name, "")
                if not os.path.exists(path):
                    continue
                if name == "xgboost":
                    m = xgb.XGBClassifier()
                    m.load_model(path)
                    self.models[name] = m
                elif name == "lightgbm":
                    self.models[name] = joblib.load(path)
                elif name == "catboost":
                    m = CatBoostClassifier()
                    m.load_model(path)
                    self.models[name] = m

            if self.models:
                self.is_trained = True
                logger.info(
                    "Ensemble loaded: %s",
                    ", ".join(self.models.keys()),
                )
        except Exception as exc:
            logger.warning("Could not load ensemble: %s", exc)
            self.is_trained = False
