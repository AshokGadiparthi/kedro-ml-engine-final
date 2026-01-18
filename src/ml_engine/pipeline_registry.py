"""
Project pipelines - COMPLETE ALL 4 PHASES INTEGRATED
================================================================================
Phase 1: Data Loading, Validation, Cleaning ✅
Phase 2: Feature Engineering & Selection ✅
Phase 3: Model Training & Evaluation ✅ (NEW)
Phase 4: Complete ML Algorithms (50+) ✅ (NEW)
================================================================================
"""

from typing import Dict
from kedro.pipeline import Pipeline

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING, VALIDATION, CLEANING
# ════════════════════════════════════════════════════════════════════════════

from ml_engine.pipelines.data_loading import create_pipeline as create_data_loading_pipeline
from ml_engine.pipelines.data_validation import create_pipeline as create_data_validation_pipeline
from ml_engine.pipelines.data_cleaning import create_pipeline as create_data_cleaning_pipeline

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2: FEATURE ENGINEERING & SELECTION (PERFECT)
# ════════════════════════════════════════════════════════════════════════════

from ml_engine.pipelines.feature_engineering import create_pipeline as create_feature_engineering_pipeline
from ml_engine.pipelines.feature_selection import create_pipeline as create_feature_selection_pipeline

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3: MODEL TRAINING & EVALUATION (NEW)
# ════════════════════════════════════════════════════════════════════════════

try:
    from ml_engine.pipelines.model_training import create_pipeline as create_model_training_pipeline
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False
    create_model_training_pipeline = None

# ════════════════════════════════════════════════════════════════════════════
# PHASE 4: COMPLETE ML ALGORITHMS (50+) (NEW)
# ════════════════════════════════════════════════════════════════════════════

try:
    from ml_engine.pipelines.phase4_algorithms import create_pipeline as create_phase4_pipeline
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False
    create_phase4_pipeline = None


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Register the project's pipelines (ALL 4 PHASES).

    Kedro calls this function to get all available pipelines.

    Data Flow:
      Phase 1: raw_data → cleaned_data + X_train_raw, X_test_raw, y_train, y_test
      Phase 2: → X_train_selected, X_test_selected (engineered & selected features)
      Phase 3: → best_model, model_evaluation (trained model + metrics)
      Phase 4: → 50+ trained models, comparison report (all algorithms)

    Returns:
        A mapping from pipeline name to Pipeline object.
    """

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA LOADING, VALIDATION, CLEANING
    # ════════════════════════════════════════════════════════════════════════

    data_loading_pipeline = create_data_loading_pipeline()
    data_validation_pipeline = create_data_validation_pipeline()
    data_cleaning_pipeline = create_data_cleaning_pipeline()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: FEATURE ENGINEERING & SELECTION
    # ════════════════════════════════════════════════════════════════════════

    feature_engineering_pipeline = create_feature_engineering_pipeline()
    feature_selection_pipeline = create_feature_selection_pipeline()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3: MODEL TRAINING & EVALUATION (NEW)
    # ════════════════════════════════════════════════════════════════════════

    if PHASE3_AVAILABLE:
        model_training_pipeline = create_model_training_pipeline()
    else:
        model_training_pipeline = None

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4: COMPLETE ML ALGORITHMS (50+) (NEW)
    # ════════════════════════════════════════════════════════════════════════

    if PHASE4_AVAILABLE:
        phase4_algorithms_pipeline = create_phase4_pipeline()
    else:
        phase4_algorithms_pipeline = None

    # ════════════════════════════════════════════════════════════════════════
    # REGISTER ALL PIPELINES
    # ════════════════════════════════════════════════════════════════════════

    pipelines = {
        # ────────────────────────────────────────────────────────────────────
        # INDIVIDUAL PHASES (for selective running)
        # ────────────────────────────────────────────────────────────────────

        "data_loading": data_loading_pipeline,
        "data_validation": data_validation_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "feature_selection": feature_selection_pipeline,

        # ────────────────────────────────────────────────────────────────────
        # PHASE COMBINATIONS
        # ────────────────────────────────────────────────────────────────────

        # Phase 1: Data Prep
        "phase1": (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline
        ),

        # Phase 2: Feature Engineering
        "phase2": (
                feature_engineering_pipeline +
                feature_selection_pipeline
        ),

        # Phase 1 + 2: Complete Data Processing
        "phase1_2": (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline
        ),

        "data_processing": (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline
        ),

        # Default pipeline (Phase 1 only)
        "__default__": (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline
        ),
    }

    # ════════════════════════════════════════════════════════════════════════
    # ADD PHASE 3 IF AVAILABLE
    # ════════════════════════════════════════════════════════════════════════

    if PHASE3_AVAILABLE and model_training_pipeline:
        pipelines["phase3"] = model_training_pipeline

        pipelines["model_training"] = model_training_pipeline

        pipelines["phase1_2_3"] = (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline +
                model_training_pipeline
        )

    # ════════════════════════════════════════════════════════════════════════
    # ADD PHASE 4 IF AVAILABLE
    # ════════════════════════════════════════════════════════════════════════

    if PHASE4_AVAILABLE and phase4_algorithms_pipeline:
        pipelines["phase4"] = phase4_algorithms_pipeline

        pipelines["algorithms"] = phase4_algorithms_pipeline

        # Phase 3 + 4: Complete Model Training
        if PHASE3_AVAILABLE and model_training_pipeline:
            pipelines["phase3_4"] = (
                    model_training_pipeline +
                    phase4_algorithms_pipeline
            )

            pipelines["model_training_complete"] = (
                    model_training_pipeline +
                    phase4_algorithms_pipeline
            )

    # ════════════════════════════════════════════════════════════════════════
    # COMPLETE PIPELINES (ALL 4 PHASES)
    # ════════════════════════════════════════════════════════════════════════

    if PHASE3_AVAILABLE and PHASE4_AVAILABLE and model_training_pipeline and phase4_algorithms_pipeline:
        # All 4 phases together
        pipelines["phase1_2_3_4"] = (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline +
                model_training_pipeline +
                phase4_algorithms_pipeline
        )

        pipelines["complete"] = (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline +
                model_training_pipeline +
                phase4_algorithms_pipeline
        )

        pipelines["all"] = (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline +
                model_training_pipeline +
                phase4_algorithms_pipeline
        )

    return pipelines