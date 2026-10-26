"""
ML Model Loader for Smart Farming Assistant.
Loads trained Joblib model artifacts from backend/models/.
"""
import logging
from pathlib import Path
from typing import Any, Dict
import joblib

logger = logging.getLogger(__name__)

# Base directory relative to backend application: .../backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Exact model filenames specification
MODEL_FILES: Dict[str, str] = {
    "disease_model": "disease_risk_model.joblib",
    "pest_model": "pest_risk_model.joblib",
    "nutrient_model": "nutrient_deficiency_model.joblib",
    "irrigation_model": "irrigation_model.joblib",
}

# Module-level references for exposed models
disease_model: Any = None
pest_model: Any = None
nutrient_model: Any = None
irrigation_model: Any = None

_models_loaded: bool = False


def get_model_path(filename: str) -> Path:
    """Resolves relative model path against backend/models directory."""
    return MODELS_DIR / filename


def verify_model_artifacts_exist() -> None:
    """
    Validates presence of all four required model files in backend/models.
    Raises FileNotFoundError with exact missing filenames if any are absent.
    """
    missing_files: list[str] = []
    for model_key, filename in MODEL_FILES.items():
        target_path = get_model_path(filename)
        if not target_path.is_file():
            missing_files.append(f"{filename} (expected for {model_key})")

    if missing_files:
        missing_formatted = "\n  - " + "\n  - ".join(missing_files)
        raise FileNotFoundError(
            f"Startup Error: Missing required trained model artifact(s) in '{MODELS_DIR}':{missing_formatted}\n"
            f"Ensure all four .joblib files are placed in backend/models/."
        )


def load_all_models() -> Dict[str, Any]:
    """
    Loads all four .joblib models, verifies that each supports predict(),
    and exposes them via module-level variables.
    """
    global disease_model, pest_model, nutrient_model, irrigation_model, _models_loaded

    # 1. Check all four files exist before loading
    verify_model_artifacts_exist()

    loaded_instances: Dict[str, Any] = {}

    for key, filename in MODEL_FILES.items():
        path = get_model_path(filename)
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to unpickle model '{key}' from '{path}': {exc}"
            ) from exc

        # 2. Simple verification that loaded object supports predict()
        if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
            raise TypeError(
                f"Model validation error: Loaded object '{key}' from '{filename}' "
                f"of type {type(model)} does not support predict()."
            )

        loaded_instances[key] = model

    # 3. Expose the four loaded models
    disease_model = loaded_instances["disease_model"]
    pest_model = loaded_instances["pest_model"]
    nutrient_model = loaded_instances["nutrient_model"]
    irrigation_model = loaded_instances["irrigation_model"]
    _models_loaded = True

    return loaded_instances


def is_loaded() -> bool:
    """Checks if all models are loaded."""
    return _models_loaded
