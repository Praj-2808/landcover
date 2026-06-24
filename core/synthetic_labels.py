"""
core/synthetic_labels.py
Generates synthetic, threshold-based training samples for land-cover
classification when no ground-truth labels are available.

Approach:
  - For each pixel in a feature stack, apply well-established spectral
    index thresholds (NDVI, NDWI, NDBI, brightness) to assign a
    pseudo-label.
  - Sample a balanced subset of pseudo-labeled pixels per class to use
    as training data for supervised models (Random Forest, XGBoost,
    LightGBM).

This is a heuristic bootstrap approach: it encodes domain knowledge about
spectral signatures of water, vegetation, urban, agriculture, and bare
land into rule-based labels, then lets ML models learn a richer decision
boundary across the full feature space (raw bands + indices).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from config import (
    CLASS_NAME_TO_CODE,
    FEATURE_NAMES,
    N_SYNTHETIC_SAMPLES_PER_CLASS,
    RANDOM_STATE,
)
from core.features import FeatureStack
from utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticLabelGenerator:
    """Generates rule-based pseudo-labels and balanced training samples."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self._rng = np.random.default_rng(random_state)

    def generate_pseudo_labels(self, feature_stack: FeatureStack) -> np.ndarray:
        """
        Apply spectral index thresholds to assign a land-cover class to
        every pixel.

        Args:
            feature_stack: FeatureStack with computed indices.

        Returns:
            A 2D int array (H, W) of class codes (see config.LAND_COVER_CLASSES).
        """
        ndvi = feature_stack.arrays["ndvi"]
        ndwi = feature_stack.arrays["ndwi"]
        ndbi = feature_stack.arrays["ndbi"]
        savi = feature_stack.arrays["savi"]
        red = feature_stack.arrays["red"]
        green = feature_stack.arrays["green"]
        blue = feature_stack.arrays["blue"]

        brightness = (red + green + blue) / 3.0

        h, w = ndvi.shape
        labels = np.full((h, w), CLASS_NAME_TO_CODE["Bare Land"], dtype=np.int32)

        water_mask = ndwi > 0.0
        urban_mask = (~water_mask) & (ndbi > 0.0) & (ndvi < 0.3)
        dense_veg_mask = (~water_mask) & (ndvi > 0.4)
        agri_mask = (
            (~water_mask)
            & (~urban_mask)
            & (~dense_veg_mask)
            & (ndvi >= 0.15)
            & (ndvi <= 0.4)
            & (savi > 0.1)
        )
        bare_mask = (
            (~water_mask)
            & (~urban_mask)
            & (~dense_veg_mask)
            & (~agri_mask)
            & (ndvi < 0.15)
            & (brightness > 0.15)
        )

        labels[water_mask] = CLASS_NAME_TO_CODE["Water"]
        labels[dense_veg_mask] = CLASS_NAME_TO_CODE["Vegetation"]
        labels[agri_mask] = CLASS_NAME_TO_CODE["Agriculture"]
        labels[urban_mask] = CLASS_NAME_TO_CODE["Urban/Built-up"]
        labels[bare_mask] = CLASS_NAME_TO_CODE["Bare Land"]
        # Anything not covered defaults to Vegetation if NDVI moderate else Bare Land
        remaining = ~(water_mask | urban_mask | dense_veg_mask | agri_mask | bare_mask)
        labels[remaining & (ndvi >= 0.15)] = CLASS_NAME_TO_CODE["Vegetation"]
        labels[remaining & (ndvi < 0.15)] = CLASS_NAME_TO_CODE["Bare Land"]

        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Pseudo-label distribution: %s", dict(zip(unique.tolist(), counts.tolist())))

        return labels

    def sample_training_data(
        self, feature_stack: FeatureStack, pseudo_labels: np.ndarray,
        samples_per_class: int = N_SYNTHETIC_SAMPLES_PER_CLASS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw a balanced random sample of pixels per class for training.

        Args:
            feature_stack: FeatureStack with computed features.
            pseudo_labels: 2D int array of pseudo-labels (same shape as features).
            samples_per_class: Max number of samples to draw per class.

        Returns:
            Tuple of (X, y): feature matrix (N, F) and label vector (N,).
        """
        feature_matrix = feature_stack.to_pixel_matrix()
        labels_flat = pseudo_labels.reshape(-1)

        X_parts = []
        y_parts = []
        for class_code in sorted(np.unique(labels_flat)):
            idx = np.where(labels_flat == class_code)[0]
            if idx.size == 0:
                continue
            n = min(samples_per_class, idx.size)
            chosen = self._rng.choice(idx, size=n, replace=False)
            X_parts.append(feature_matrix[chosen])
            y_parts.append(labels_flat[chosen])

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)

        # Remove rows with NaN/Inf
        finite_mask = np.isfinite(X).all(axis=1)
        X, y = X[finite_mask], y[finite_mask]

        logger.info("Sampled %d training pixels across %d classes (features=%d).",
                    X.shape[0], len(np.unique(y)), X.shape[1])
        return X, y
