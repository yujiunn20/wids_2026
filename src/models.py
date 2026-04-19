from lightgbm import LGBMClassifier


def get_lgbm_model():
    return LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )