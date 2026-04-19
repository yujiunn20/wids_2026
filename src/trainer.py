import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

from src.models import get_lgbm_model


def run_cv_training(X, X_test, targets, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_scores = {k: [] for k in targets}
    test_preds = {k: np.zeros(len(X_test)) for k in targets}

    split_target = targets["prob_48h"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, split_target), 1):
        print(f"\n===== Fold {fold} =====")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        for name, y in targets.items():
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = get_lgbm_model()
            model.fit(X_train, y_train)

            val_pred = model.predict_proba(X_val)[:, 1]
            score = brier_score_loss(y_val, val_pred)
            oof_scores[name].append(score)

            test_pred = model.predict_proba(X_test)[:, 1]
            test_preds[name] += test_pred / n_splits

            print(f"{name}: {score:.5f}")

    print("\n===== Mean CV Brier Scores =====")
    for name, scores in oof_scores.items():
        print(f"{name}: {np.mean(scores):.5f}")

    return test_preds, oof_scores