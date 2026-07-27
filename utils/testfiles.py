from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

files = [
    "best15_xgb.pkl",
    "card_database.pkl",
    "category_dict.pkl",
]

for filename in files:
    path = MODELS_DIR / filename

    print(f"\nTesting: {path}")

    try:
        item = joblib.load(path)
        print("Loaded successfully")
        print("Type:", type(item))

    except Exception as error:
        print("FAILED")
        print(type(error).__name__, error)