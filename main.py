import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss


def main():
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")

    feature_cols = [
        "num_perimeters_0_5h",
        "dt_first_last_0_5h",
        "area_first_ha",
        "area_growth_abs_0_5h",
        "area_growth_rate_ha_per_h",
        "radial_growth_m",
        "radial_growth_rate_m_per_h",
        "area_growth_rel_0_5h",
        "log1p_area_first",
        "log1p_growth",
        "log_area_ratio_0_5h",
        "centroid_speed_m_per_h",
        "dist_min_ci_0_5h",
        "closing_speed_m_per_h",
        "dist_slope_ci_0_5h",
        "dist_change_ci_0_5h",
        "dist_std_ci_0_5h",
        "dist_fit_r2_0_5h",
        "alignment_cos",
        "along_track_speed",
        "event_start_hour",
        "event_start_dayofweek",
        "event_start_month",
    ]

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    y_12 = (train["time_to_hit_hours"] <= 12).astype(int)
    y_24 = (train["time_to_hit_hours"] <= 24).astype(int)
    y_48 = (train["time_to_hit_hours"] <= 48).astype(int)
    y_72 = train["event"].astype(int)

    targets = {
        "prob_12h": y_12,
        "prob_24h": y_24,
        "prob_48h": y_48,
        "prob_72h": y_72,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_scores = {k: [] for k in targets}
    test_preds = {k: np.zeros(len(X_test)) for k in targets}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_48), 1):
        print(f"\n===== Fold {fold} =====")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        for name, y in targets.items():
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = LGBMClassifier(
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

            model.fit(X_train, y_train)

            val_pred = model.predict_proba(X_val)[:, 1]
            score = brier_score_loss(y_val, val_pred)
            oof_scores[name].append(score)

            test_pred = model.predict_proba(X_test)[:, 1]
            test_preds[name] += test_pred / skf.n_splits

            print(f"{name}: {score:.5f}")

    print("\n===== Mean CV Brier Scores =====")
    for name, scores in oof_scores.items():
        print(f"{name}: {np.mean(scores):.5f}")

    pred_array = np.column_stack([
        test_preds["prob_12h"],
        test_preds["prob_24h"],
        test_preds["prob_48h"],
        test_preds["prob_72h"],
    ])

    pred_array = np.maximum.accumulate(pred_array, axis=1)

    submission = pd.DataFrame({
        "event_id": test["event_id"],
        "prob_12h": pred_array[:, 0],
        "prob_24h": pred_array[:, 1],
        "prob_48h": pred_array[:, 2],
        "prob_72h": pred_array[:, 3],
    })

    submission.to_csv("data/submissions/submission_kfold.csv", index=False)
    print("\nSaved to data/submissions/submission_kfold.csv")


if __name__ == "__main__":
    main()