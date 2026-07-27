import joblib
import shap
import streamlit as st
from xgboost import XGBClassifier

from utils.constants import (
    CARD_DATABASE_PATH,
    CATEGORY_DICT_PATH,
    MODEL_INFO_PATH,
    MODEL_PATH,
)


# ===================================================
# LOAD MODEL AND DICTIONARIES
# ===================================================

@st.cache_resource
def load_resources():
    """Load the model, feature order, and dictionaries."""

    model = XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    model_info = joblib.load(
        MODEL_INFO_PATH
    )

    if not isinstance(model_info, dict):
        raise ValueError(
            "model_info.pkl must contain a dictionary."
        )

    if "features" not in model_info:
        raise ValueError(
            "model_info.pkl is missing the "
            "'features' key."
        )

    model_features = model_info[
        "features"
    ]

    if not model_features:
        raise ValueError(
            "The saved feature list is empty."
        )

    card_database = joblib.load(
        CARD_DATABASE_PATH
    )

    category_dict = joblib.load(
        CATEGORY_DICT_PATH
    )

    if not isinstance(card_database, dict):
        raise ValueError(
            "card_database.pkl must contain "
            "a dictionary."
        )

    if not isinstance(category_dict, dict):
        raise ValueError(
            "category_dict.pkl must contain "
            "a dictionary."
        )

    return (
        model,
        model_features,
        card_database,
        category_dict,
    )


# ===================================================
# LOAD SHAP EXPLAINER
# ===================================================

@st.cache_resource
def load_shap_explainer(_model):
    """Create and cache a SHAP explainer."""

    return shap.TreeExplainer(
        _model
    )