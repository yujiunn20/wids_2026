import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

from src.models import get_model
from src.calibration import fit_isotonic_calibrator, apply_calibrator


def run_cv_training(X, X_test, targets, model_name="lgbm", n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_scores = {k: [] for k in targets}
    test_preds = {k: np.zeros(len(X_test)) for k in targets}

    split_target = targets["prob_72h"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, split_target), 1):
        print(f"\n===== Fold {fold} =====")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        for name, y in targets.items():
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = get_model(model_name)
            model.fit(X_train, y_train)

            val_pred_raw = model.predict_proba(X_val)[:, 1]
            test_pred_raw = model.predict_proba(X_test)[:, 1]

            calibrator = fit_isotonic_calibrator(y_val, val_pred_raw)
            val_pred = apply_calibrator(calibrator, val_pred_raw)
            test_pred = apply_calibrator(calibrator, test_pred_raw)

            score = brier_score_loss(y_val, val_pred)
            oof_scores[name].append(score)
            test_preds[name] += test_pred / n_splits

            print(f"{name}: {score:.5f}")

    print(f"\n===== Mean CV Brier Scores ({model_name}) =====")
    for name, scores in oof_scores.items():
        print(f"{name}: {np.mean(scores):.5f}")

    return test_preds, oof_scores

def run_cv_training_dual(X, X_test, targets, model_name="lgbm", n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_scores = {k: [] for k in targets}
    test_preds = {k: np.zeros(len(X_test)) for k in targets}

    split_target = targets["prob_72h"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, split_target), 1):
        print(f"\n===== Fold {fold} =====")

        X_train_full = X.iloc[train_idx]
        X_val_full = X.iloc[val_idx]

        single_train_mask = X_train_full["num_perimeters_0_5h"] == 1
        multi_train_mask = X_train_full["num_perimeters_0_5h"] > 1

        single_val_mask = X_val_full["num_perimeters_0_5h"] == 1
        multi_val_mask = X_val_full["num_perimeters_0_5h"] > 1

        single_test_mask = X_test["num_perimeters_0_5h"] == 1
        multi_test_mask = X_test["num_perimeters_0_5h"] > 1

        for name, y in targets.items():
            y_train_full = y.iloc[train_idx]
            y_val_full = y.iloc[val_idx]

            val_pred_full = np.zeros(len(X_val_full))
            test_pred_full = np.zeros(len(X_test))

            # -------------------------
            # 1) single model
            # -------------------------
            if single_train_mask.sum() > 1 and single_val_mask.sum() > 0:
                X_train_single = X_train_full.loc[single_train_mask]
                y_train_single = y_train_full.loc[single_train_mask]

                X_val_single = X_val_full.loc[single_val_mask]
                X_test_single = X_test.loc[single_test_mask]

                model_single = get_model(model_name)
                model_single.fit(X_train_single, y_train_single)

                val_pred_raw_single = model_single.predict_proba(X_val_single)[:, 1]
                test_pred_raw_single = model_single.predict_proba(X_test_single)[:, 1]

                calibrator_single = fit_isotonic_calibrator(
                    y_val_full.loc[single_val_mask],
                    val_pred_raw_single
                )
                val_pred_single = apply_calibrator(calibrator_single, val_pred_raw_single)
                test_pred_single = apply_calibrator(calibrator_single, test_pred_raw_single)

                val_pred_full[single_val_mask.values] = val_pred_single
                test_pred_full[single_test_mask.values] = test_pred_single

            # -------------------------
            # 2) multi model
            # -------------------------
            if multi_train_mask.sum() > 1 and multi_val_mask.sum() > 0:
                X_train_multi = X_train_full.loc[multi_train_mask]
                y_train_multi = y_train_full.loc[multi_train_mask]

                X_val_multi = X_val_full.loc[multi_val_mask]
                X_test_multi = X_test.loc[multi_test_mask]

                model_multi = get_model(model_name)
                model_multi.fit(X_train_multi, y_train_multi)

                val_pred_raw_multi = model_multi.predict_proba(X_val_multi)[:, 1]
                test_pred_raw_multi = model_multi.predict_proba(X_test_multi)[:, 1]

                calibrator_multi = fit_isotonic_calibrator(
                    y_val_full.loc[multi_val_mask],
                    val_pred_raw_multi
                )
                val_pred_multi = apply_calibrator(calibrator_multi, val_pred_raw_multi)
                test_pred_multi = apply_calibrator(calibrator_multi, test_pred_raw_multi)

                val_pred_full[multi_val_mask.values] = val_pred_multi
                test_pred_full[multi_test_mask.values] = test_pred_multi

            score = brier_score_loss(y_val_full, val_pred_full)
            oof_scores[name].append(score)
            test_preds[name] += test_pred_full / n_splits

            print(f"{name}: {score:.5f}")

    print(f"\n===== Mean CV Brier Scores ({model_name}, dual) =====")
    for name, scores in oof_scores.items():
        print(f"{name}: {np.mean(scores):.5f}")

    return test_preds, oof_scores