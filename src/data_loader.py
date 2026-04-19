import pandas as pd


def load_data(train_path="data/raw/train.csv", test_path="data/raw/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test