import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier


def main():
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")

    # 不可拿來當 feature 的欄位
    drop_cols = ["event_id", "time_to_hit_hours", "event"]

    # 只保留 train / test 共同擁有的 feature
    feature_cols = [col for col in train.columns if col not in drop_cols and col in test.columns]

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # 四個 target
    y_12 = (train["time_to_hit_hours"] <= 12).astype(int)
    y_24 = (train["time_to_hit_hours"] <= 24).astype(int)
    y_48 = (train["time_to_hit_hours"] <= 48).astype(int)
    y_72 = (train["time_to_hit_hours"] <= 72).astype(int)

    preds = {}

    targets = {
        "prob_12h": y_12,
        "prob_24h": y_24,
        "prob_48h": y_48,
        "prob_72h": y_72,
    }

    for name, y in targets.items():
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        preds[name] = model.predict_proba(X_test)[:, 1]

    # stack 成 array
    pred_array = np.column_stack([
        preds["prob_12h"],
        preds["prob_24h"],
        preds["prob_48h"],
        preds["prob_72h"],
    ])
    
    # 強制單調遞增
    pred_array = np.maximum.accumulate(pred_array, axis=1)

    submission = pd.DataFrame({
        "event_id": test["event_id"],
        "prob_12h": pred_array[:, 0],
        "prob_24h": pred_array[:, 1],
        "prob_48h": pred_array[:, 2],
        "prob_72h": pred_array[:, 3],
    })

    submission.to_csv("data/submissions/submission_baseline.csv", index=False)

    print("X shape:", X.shape)
    print("X_test shape:", X_test.shape)
    print("Submission saved to data/submissions/submission_baseline.csv")


if __name__ == "__main__":
    main()