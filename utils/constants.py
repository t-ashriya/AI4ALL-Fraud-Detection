from pathlib import Path


# ===================================================
# APPLICATION SETTINGS
# ===================================================

PREDICTION_THRESHOLD = 0.80


# ===================================================
# REQUIRED CSV COLUMNS
# ===================================================

REQUIRED_COLUMNS = [
    "amt",
    "trans_date_trans_time",
    "cc_num",
    "category",
    "unix_time",
    "dob",
]


# ===================================================
# FILE PATHS
# ===================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best15_xgb2.json"
MODEL_INFO_PATH = MODELS_DIR / "model_info.pkl"
CARD_DATABASE_PATH = MODELS_DIR / "card_database.pkl"
CATEGORY_DICT_PATH = MODELS_DIR / "category_dict.pkl"