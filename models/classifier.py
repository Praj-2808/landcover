"""
models/classifier.py
Land-cover classification models: Random Forest (primary), XGBoost
(secondary), and LightGBM (optional comparison). Provides a unified
interface for training, saving, loading, evaluating, and predicting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from config import (
    FEATURE_NAMES,
    LAND_COVER_CLASSES,
    LIGHTGBM_PARAMS,
    MODEL_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    XGB_PARAMS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_MODEL_TYPES = ("random_forest", "xgboost", "lightgbm")


@dataclass
class EvaluationResult:
    """Holds model evaluation metrics."""

    accuracy: float
    f1_macro: float
    confusion: np.ndarray
    report: Dict

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion.tolist(),
            "report": self.report,
        }


class LandCoverClassifier:
    """
    Unified wrapper around supervised classifiers used for land-cover
    classification.

    Supported model types: 'random_forest', 'xgboost', 'lightgbm'.
    """

    def __init__(self, model_type: str = "random_forest") -> None:
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Must be one of {VALID_MODEL_TYPES}."
            )
        self.model_type = model_type
        self._model = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(**RF_PARAMS)

        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "xgboost is not installed. Install it with `pip install xgboost`."
                ) from exc
            params = dict(XGB_PARAMS)
            return XGBClassifier(**params)

        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError(
                    "lightgbm is not installed. Install it with `pip install lightgbm`."
                ) from exc
            return LGBMClassifier(**LIGHTGBM_PARAMS)

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    def train(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
    ) -> EvaluationResult:
        """
        Train the model on (X, y), holding out a test split for evaluation.

        Args:
            X: Feature matrix (N, F).
            y: Label vector (N,).
            test_size: Fraction of data held out for evaluation.

        Returns:
            EvaluationResult with accuracy, macro F1, confusion matrix, and
            a per-class classification report.
        """
        logger.info(
            "Training %s on %d samples (%d features), test_size=%.2f",
            self.model_type, X.shape[0], X.shape[1], test_size,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y,
        )

        self._model = self._build_model()

        if self.model_type == "xgboost":
            # XGBoost requires contiguous 0..K-1 labels
            classes = sorted(np.unique(y))
            label_map = {c: i for i, c in enumerate(classes)}
            inv_map = {i: c for c, i in label_map.items()}
            y_train_mapped = np.array([label_map[v] for v in y_train])
            self._model.fit(X_train, y_train_mapped)
            y_pred_mapped = self._model.predict(X_test)
            y_pred = np.array([inv_map[v] for v in y_pred_mapped])
            self._label_map = label_map
            self._inv_label_map = inv_map
        elif self.model_type == "lightgbm":
            classes = sorted(np.unique(y))
            label_map = {c: i for i, c in enumerate(classes)}
            inv_map = {i: c for c, i in label_map.items()}
            y_train_mapped = np.array([label_map[v] for v in y_train])
            self._model.fit(X_train, y_train_mapped)
            y_pred_mapped = self._model.predict(X_test)
            y_pred = np.array([inv_map[v] for v in y_pred_mapped])
            self._label_map = label_map
            self._inv_label_map = inv_map
        else:
            self._model.fit(X_train, y_train)
            y_pred = self._model.predict(X_test)
            self._label_map = None
            self._inv_label_map = None

        self._is_fitted = True

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=sorted(LAND_COVER_CLASSES.keys()))
        report = classification_report(
            y_test, y_pred,
            labels=sorted(LAND_COVER_CLASSES.keys()),
            target_names=[LAND_COVER_CLASSES[k] for k in sorted(LAND_COVER_CLASSES.keys())],
            output_dict=True,
            zero_division=0,
        )

        logger.info(
            "Training complete: accuracy=%.4f, macro-F1=%.4f", accuracy, f1_macro,
        )

        return EvaluationResult(accuracy=accuracy, f1_macro=f1_macro, confusion=cm, report=report)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> EvaluationResult:
        """Evaluate the fitted model on a held-out dataset."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained or loaded.")

        y_pred = self.predict_labels(X)
        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        cm = confusion_matrix(y, y_pred, labels=sorted(LAND_COVER_CLASSES.keys()))
        report = classification_report(
            y, y_pred,
            labels=sorted(LAND_COVER_CLASSES.keys()),
            target_names=[LAND_COVER_CLASSES[k] for k in sorted(LAND_COVER_CLASSES.keys())],
            output_dict=True,
            zero_division=0,
        )
        return EvaluationResult(accuracy=accuracy, f1_macro=f1_macro, confusion=cm, report=report)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Predict land-cover class codes for a feature matrix.

        Args:
            X: Feature matrix (N, F).

        Returns:
            1D array of predicted class codes (N,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained or loaded.")

        if self.model_type in ("xgboost", "lightgbm") and self._inv_label_map is not None:
            pred_mapped = self._model.predict(X)
            return np.array([self._inv_label_map[int(v)] for v in pred_mapped])

        return self._model.predict(X)

    def predict_map(self, X: np.ndarray, height: int, width: int) -> np.ndarray:
        """
        Predict a 2D land-cover classification map.

        Args:
            X: Feature matrix (H*W, F).
            height: Output map height.
            width: Output map width.

        Returns:
            2D int array (H, W) of class codes.
        """
        # Handle non-finite values defensively
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds = self.predict_labels(X_clean)
        return preds.reshape(height, width).astype(np.int32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, name: str) -> Path:
        """
        Save the trained model and metadata to disk.

        Args:
            name: Base filename (without extension), e.g. 'mumbai_2025'.

        Returns:
            Path to the saved model file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an untrained model.")

        path = Path(MODEL_DIR) / f"{name}_{self.model_type}.joblib"
        payload = {
            "model": self._model,
            "model_type": self.model_type,
            "label_map": getattr(self, "_label_map", None),
            "inv_label_map": getattr(self, "_inv_label_map", None),
            "feature_names": FEATURE_NAMES,
        }
        joblib.dump(payload, path)
        logger.info("Saved model to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "LandCoverClassifier":
        """
        Load a previously saved model.

        Args:
            path: Path to the .joblib model file.

        Returns:
            A fitted LandCoverClassifier instance.
        """
        payload = joblib.load(path)
        instance = cls(model_type=payload["model_type"])
        instance._model = payload["model"]
        instance._label_map = payload.get("label_map")
        instance._inv_label_map = payload.get("inv_label_map")
        instance._is_fitted = True
        logger.info("Loaded model from %s", path)
        return instance

    @property
    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Return feature importances as a name->value dict, if available."""
        if not self._is_fitted:
            return None
        importances = getattr(self._model, "feature_importances_", None)
        if importances is None:
            return None
        return dict(zip(FEATURE_NAMES, importances.tolist()))
