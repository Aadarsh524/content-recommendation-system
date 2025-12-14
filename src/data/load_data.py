import pickle
import pandas as pd
from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    VAL_PATH,
    USER_ITEM_MAP_PATH
)

def load_train_val_test():
    train = pd.read_pickle(TRAIN_PATH)
    val   = pd.read_pickle(VAL_PATH)
    test  = pd.read_pickle(TEST_PATH)
    return train, val, test

def load_user_item_map():
    with open(USER_ITEM_MAP_PATH, "rb") as f:
        return pickle.load(f)