"""Pydantic schemas for Smart Farming Assistant tabular ML features."""
from typing import Literal
from pydantic import BaseModel, Field


class FarmingFeaturesRequest(BaseModel):
    """
    Exact 23 tabular input features for the Smart Farming ML system.
    """
    # 1-3: Crop Profile
    crop_type: Literal[
        "Rice", "Wheat", "Maize", "Tomato", "Potato",
        "Cotton", "Chilli", "Groundnut", "Onion", "Sugarcane"
    ] = Field(..., description="Crop type or species", example="Tomato")
    crop_age_days: int = Field(..., ge=1, description="Crop age in days", example=45)
    growth_stage: Literal[
        "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"
    ] = Field(..., description="Current crop growth stage", example="Vegetative")

    # 4-7: Weather & Climate
    temperature_c: float = Field(..., description="Ambient temperature in °C", example=26.5)
    humidity_percent: float = Field(..., ge=0.0, le=100.0, description="Relative humidity percentage", example=72.0)
    rainfall_mm: float = Field(..., ge=0.0, description="Rainfall in mm", example=120.5)
    sunlight_hours: float = Field(..., ge=0.0, le=24.0, description="Sunlight exposure in hours per day", example=7.5)

    # 8-13: Soil Chemistry & Moisture
    soil_type: Literal[
        "Sandy", "Loamy", "Clay", "Silty", "Sandy Loam"
    ] = Field(..., description="Soil texture classification", example="Loamy")
    soil_moisture_percent: float = Field(..., ge=0.0, le=100.0, description="Soil moisture percentage", example=45.0)
    soil_ph: float = Field(..., ge=0.0, le=14.0, description="Soil pH level", example=6.5)
    nitrogen_ppm: float = Field(..., ge=0.0, description="Soil nitrogen content in ppm", example=85.0)
    phosphorus_ppm: float = Field(..., ge=0.0, description="Soil phosphorus content in ppm", example=42.0)
    potassium_ppm: float = Field(..., ge=0.0, description="Soil potassium content in ppm", example=50.0)

    # 14-19: Binary Symptom Indicators (0 = No, 1 = Yes)
    leaf_yellowing: Literal[0, 1] = Field(..., description="Leaf yellowing symptom (0=No, 1=Yes)", example=0)
    leaf_spots: Literal[0, 1] = Field(..., description="Leaf spots symptom (0=No, 1=Yes)", example=1)
    leaf_curling: Literal[0, 1] = Field(..., description="Leaf curling symptom (0=No, 1=Yes)", example=0)
    wilting: Literal[0, 1] = Field(..., description="Plant wilting symptom (0=No, 1=Yes)", example=0)
    stunted_growth: Literal[0, 1] = Field(..., description="Stunted plant growth (0=No, 1=Yes)", example=0)
    pest_signs: Literal[0, 1] = Field(..., description="Visible signs of pests (0=No, 1=Yes)", example=1)

    # 20-21: Pathology & Pest Severity
    leaf_damage_percent: float = Field(..., ge=0.0, le=100.0, description="Percentage of leaf surface damaged", example=15.0)
    insect_count_per_plant: int = Field(..., ge=0, description="Insect count observed per plant", example=4)

    # 22-23: Irrigation History
    days_since_last_irrigation: int = Field(..., ge=0, description="Days elapsed since last irrigation", example=3)
    previous_irrigation_liters: float = Field(..., ge=0.0, description="Volume of water applied in previous irrigation in liters", example=250.0)


class PredictionResponse(BaseModel):
    """Output prediction response with exactly the four target fields."""
    disease_risk: str = Field(..., description="Predicted disease risk category")
    pest_risk: str = Field(..., description="Predicted pest risk category")
    nutrient_deficiency: str = Field(..., description="Predicted nutrient deficiency status")
    irrigation_required: str = Field(..., description="Predicted irrigation requirement")


class FarmingFeaturesResponse(BaseModel):
    status: str
    message: str
    received_features_count: int
    ml_status: str
    features_received: dict
