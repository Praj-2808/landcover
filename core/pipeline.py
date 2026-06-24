"""
core/pipeline.py
End-to-end orchestration: AOI resolution -> imagery acquisition ->
feature engineering -> synthetic training data -> model train/load ->
classification map generation, with disk-based caching for intermediate
results to support efficient multi-year and repeated analyses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import (
    CLASSIFICATION_DIR,
    DEFAULT_MODEL_TYPE,
    MODEL_DIR,
)
from core.aoi import AreaOfInterest, AOIResolver
from core.change_detection import ChangeDetector, ChangeResult
from core.features import FeatureEngineer, FeatureStack
from core.imagery import ImageryAcquisition, ImageryResult, SceneMetadata
from core.synthetic_labels import SyntheticLabelGenerator
from models.classifier import LandCoverClassifier, EvaluationResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class YearAnalysisResult:
    """Complete analysis result for a single AOI + year."""

    aoi: AreaOfInterest
    year: int
    classification_map: np.ndarray
    feature_stack: FeatureStack
    scene: SceneMetadata
    evaluation: Optional[EvaluationResult]
    model_type: str


class LandCoverPipeline:
    """
    High-level orchestrator for single-year and multi-year land cover
    analysis. Handles caching of classification maps to avoid redundant
    downloads, training, and inference.
    """

    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE) -> None:
        self.model_type = model_type
        self.aoi_resolver = AOIResolver()
        self.imagery = ImageryAcquisition()
        self.feature_engineer = FeatureEngineer()
        self.label_generator = SyntheticLabelGenerator()
        self.change_detector = ChangeDetector()

    # ------------------------------------------------------------------
    # AOI resolution
    # ------------------------------------------------------------------
    def resolve_aoi(self, city_name: str, country: Optional[str] = None) -> AreaOfInterest:
        """Resolve a city name (and optional country) to an AreaOfInterest."""
        return self.aoi_resolver.resolve(city_name, country)

    # ------------------------------------------------------------------
    # Single-year analysis
    # ------------------------------------------------------------------
    def analyze_year(
        self, aoi: AreaOfInterest, year: int, train_new_model: bool = False,
        model_type: Optional[str] = None,
    ) -> YearAnalysisResult:
        """
        Run the full pipeline for a single AOI + year: acquire imagery,
        compute features, train (or load) a classifier, and produce a
        land-cover classification map.

        Args:
            aoi: Resolved Area of Interest.
            year: Target year for analysis.
            train_new_model: If True, force training a new model even if a
                cached model/classification exists.
            model_type: Override the default model type for this run.

        Returns:
            A YearAnalysisResult containing the classification map and
            supporting data.
        """
        model_type = model_type or self.model_type
        cache_key = self._cache_key(aoi, year, model_type)
        cache_path = Path(CLASSIFICATION_DIR) / f"{cache_key}.npz"

        if cache_path.exists() and not train_new_model:
            logger.info("Loading cached classification for %s %d (%s)", aoi.name, year, model_type)
            return self._load_cached_result(cache_path, aoi, year, model_type)

        logger.info("Running fresh analysis for %s %d using %s", aoi.name, year, model_type)

        # 1. Imagery acquisition
        imagery: ImageryResult = self.imagery.get_imagery_for_year(aoi, year)

        # 2. Feature engineering
        feature_stack = self.feature_engineer.compute_features(imagery.bands)

        # 3. Synthetic labels + training samples
        pseudo_labels = self.label_generator.generate_pseudo_labels(feature_stack)
        X_train, y_train = self.label_generator.sample_training_data(feature_stack, pseudo_labels)

        # 4. Train classifier
        classifier = LandCoverClassifier(model_type=model_type)
        evaluation = classifier.train(X_train, y_train)

        # 5. Predict full classification map
        feature_matrix = feature_stack.to_pixel_matrix()
        classification_map = classifier.predict_map(
            feature_matrix, feature_stack.height, feature_stack.width
        )

        # 6. Save model and classification cache
        model_name = f"{aoi.name.replace(' ', '_')}_{year}"
        classifier.save(model_name)
        self._save_cached_result(
            cache_path, classification_map, feature_stack, imagery.scene, evaluation,
        )

        return YearAnalysisResult(
            aoi=aoi,
            year=year,
            classification_map=classification_map,
            feature_stack=feature_stack,
            scene=imagery.scene,
            evaluation=evaluation,
            model_type=model_type,
        )

    # ------------------------------------------------------------------
    # Multi-year comparison
    # ------------------------------------------------------------------
    def compare_years(
        self, aoi: AreaOfInterest, year_from: int, year_to: int,
        model_type: Optional[str] = None,
    ) -> Dict:
        """
        Run single-year analyses for two years and compute change detection
        products between them.

        Args:
            aoi: Resolved Area of Interest.
            year_from: Earlier year.
            year_to: Later year.
            model_type: Optional model type override.

        Returns:
            Dict with 'result_from', 'result_to', and 'change' (ChangeResult).
        """
        if year_from >= year_to:
            raise ValueError("year_from must be strictly earlier than year_to.")

        result_from = self.analyze_year(aoi, year_from, model_type=model_type)
        result_to = self.analyze_year(aoi, year_to, model_type=model_type)

        change = self.change_detector.detect_change(
            result_from.classification_map, result_to.classification_map
        )

        return {"result_from": result_from, "result_to": result_to, "change": change}

    def quick_comparison_years(self, current_year: int, offset: int) -> tuple[int, int]:
        """
        Compute (year_from, year_to) for a 'Current vs N years ago' comparison.

        Args:
            current_year: The current/reference year.
            offset: Number of years to look back.

        Returns:
            Tuple of (current_year - offset, current_year).
        """
        return current_year - offset, current_year

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------
    def _cache_key(self, aoi: AreaOfInterest, year: int, model_type: str) -> str:
        raw = f"{aoi.name}_{aoi.bbox}_{year}_{model_type}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
        return f"{aoi.name.replace(' ', '_')}_{year}_{model_type}_{digest}"

    def _save_cached_result(
        self, path: Path, classification_map: np.ndarray, feature_stack: FeatureStack,
        scene: SceneMetadata, evaluation: EvaluationResult,
    ) -> None:
        try:
            feature_arrays = {f"feat_{k}": v for k, v in feature_stack.arrays.items()}
            np.savez_compressed(
                path,
                classification_map=classification_map,
                height=feature_stack.height,
                width=feature_stack.width,
                scene=np.array(json.dumps(scene.to_dict()).encode("utf-8")),
                evaluation=np.array(json.dumps(evaluation.to_dict()).encode("utf-8")),
                **feature_arrays,
            )
            logger.info("Cached classification result to %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cache classification result: %s", exc)

    def _load_cached_result(
        self, path: Path, aoi: AreaOfInterest, year: int, model_type: str,
    ) -> YearAnalysisResult:
        data = np.load(path, allow_pickle=False)
        classification_map = data["classification_map"]
        height = int(data["height"])
        width = int(data["width"])

        from config import FEATURE_NAMES
        arrays = {name: data[f"feat_{name}"] for name in FEATURE_NAMES if f"feat_{name}" in data}
        feature_stack = FeatureStack(arrays=arrays, height=height, width=width)

        scene_dict = json.loads(bytes(data["scene"]).decode("utf-8"))
        scene = SceneMetadata(**scene_dict)

        evaluation = None
        if "evaluation" in data:
            eval_dict = json.loads(bytes(data["evaluation"]).decode("utf-8"))
            evaluation = EvaluationResult(
                accuracy=eval_dict["accuracy"],
                f1_macro=eval_dict["f1_macro"],
                confusion=np.array(eval_dict["confusion_matrix"]),
                report=eval_dict["report"],
            )

        return YearAnalysisResult(
            aoi=aoi,
            year=year,
            classification_map=classification_map,
            feature_stack=feature_stack,
            scene=scene,
            evaluation=evaluation,
            model_type=model_type,
        )
