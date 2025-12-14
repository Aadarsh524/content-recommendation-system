from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Processed files
TRAIN_PATH = PROCESSED_DATA_DIR / "train.pkl"
VAL_PATH   = PROCESSED_DATA_DIR / "val.pkl"
TEST_PATH  = PROCESSED_DATA_DIR / "test.pkl"

USER_ITEM_MAP_PATH = PROCESSED_DATA_DIR / "user_item_map.pkl"

#trained model folder
TRAINED_MODEL = PROJECT_ROOT / "trained"

# Recommender defaults
TOP_K = 10
