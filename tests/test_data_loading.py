from src.data.load_data import load_train_val_test, load_user_item_map

train, val, test = load_train_val_test()
user_item_map = load_user_item_map()

print(train.shape, val.shape, test.shape)
print("Users in map:", len(user_item_map))
