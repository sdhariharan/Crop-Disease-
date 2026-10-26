"""API routes for Smart Farming Assistant (23-Feature Tabular ML System)."""
import logging
from typing import Any
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from app.core.config import settings
from app.api.schemas import FarmingFeaturesRequest, PredictionResponse
import app.ml.model_loader as ml_loader

logger = logging.getLogger(__name__)

router = APIRouter()

EXACT_23_FEATURES = [
    "crop_type",
    "crop_age_days",
    "growth_stage",
    "temperature_c",
    "humidity_percent",
    "rainfall_mm",
    "sunlight_hours",
    "soil_type",
    "soil_moisture_percent",
    "soil_ph",
    "nitrogen_ppm",
    "phosphorus_ppm",
    "potassium_ppm",
    "leaf_yellowing",
    "leaf_spots",
    "leaf_curling",
    "wilting",
    "stunted_growth",
    "pest_signs",
    "leaf_damage_percent",
    "insect_count_per_plant",
    "days_since_last_irrigation",
    "previous_irrigation_liters"
]


def format_prediction(val: Any) -> str:
    """Format single prediction scalar cleanly to string."""
    if hasattr(val, "item"):
        val = val.item()
    return str(val)


@router.get("/health")
async def health_check():
    """Health check endpoint to verify backend service status."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "system_type": "tabular_ml",
        "features_count": len(EXACT_23_FEATURES),
        "ml_engine": "loaded" if ml_loader.is_loaded() else "ready_for_integration"
    }


@router.get("/info")
async def get_system_info():
    """Returns metadata about the Smart Farming platform and the exact 23 features."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "architecture": "Tabular Machine Learning",
        "input_format": "application/json",
        "features_count": len(EXACT_23_FEATURES),
        "features_list": EXACT_23_FEATURES,
        "categorical_domains": {
            "crop_type": [
                "Rice", "Wheat", "Maize", "Tomato", "Potato",
                "Cotton", "Chilli", "Groundnut", "Onion", "Sugarcane"
            ],
            "growth_stage": [
                "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"
            ],
            "soil_type": [
                "Sandy", "Loamy", "Clay", "Silty", "Sandy Loam"
            ],
            "binary_symptoms": [
                "leaf_yellowing", "leaf_spots", "leaf_curling",
                "wilting", "stunted_growth", "pest_signs"
            ]
        },
        "ml_status": "Active (4 trained models loaded)" if ml_loader.is_loaded() else "Models staged"
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_farming_advisory(payload: FarmingFeaturesRequest):
    """
    Runs ML inference across all four trained Joblib models:
    - disease_model
    - pest_model
    - nutrient_model
    - irrigation_model
    """
    # Ensure models are loaded
    if not ml_loader.is_loaded():
        try:
            ml_loader.load_all_models()
        except Exception as load_err:
            logger.error("Failed to load models during predict request: %s", load_err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction models are not currently available."
            )

    # 1. Convert validated request into a DataFrame with exact 23 columns in exact order
    try:
        data_dict = payload.model_dump()
        input_df = pd.DataFrame([data_dict], columns=EXACT_23_FEATURES)
    except Exception as data_err:
        logger.error("Failed to assemble input DataFrame: %s", data_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing input features."
        )

    # 2. Run inference across all four trained models
    try:
        # Pipeline models receive input_df directly
        disease_pred = ml_loader.disease_model.predict(input_df)
        pest_pred = ml_loader.pest_model.predict(input_df)
        irrigation_pred = ml_loader.irrigation_model.predict(input_df)

        # Nutrient model: check if pipeline or raw estimator
        if hasattr(ml_loader.nutrient_model, "named_steps"):
            nutrient_pred = ml_loader.nutrient_model.predict(input_df)
        else:
            # Use the existing trained preprocessor from disease_model
            preprocessor = getattr(ml_loader.disease_model, "named_steps", {}).get("preprocessor")
            if preprocessor is not None:
                transformed_input = preprocessor.transform(input_df)
                nutrient_pred = ml_loader.nutrient_model.predict(transformed_input)
            else:
                nutrient_pred = ml_loader.nutrient_model.predict(input_df)

    except Exception as inference_err:
        logger.error("Inference execution failure: %s", inference_err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed while processing input features."
        )

    # 3. Return exactly the four prediction fields
    return PredictionResponse(
        disease_risk=format_prediction(disease_pred[0]),
        pest_risk=format_prediction(pest_pred[0]),
        nutrient_deficiency=format_prediction(nutrient_pred[0]),
        irrigation_required=format_prediction(irrigation_pred[0])
    )
