from src.data_loader import load_data
from src.features import (
    get_GBM_feature_columns,
    get_NN_feature_columns,
    get_GBM_feature_columns_v2,
    get_NN_feature_columns_v2,
    get_GBM_feature_columns_v3,
    get_NN_feature_columns_v3,
)
from src.targets import build_targets
from src.trainer import run_cv_training, run_cv_training_dual
from src.postprocess import enforce_monotonicity
from src.submission import save_submission
from src.ensemble import average_ensemble
from src.feature_engineering import add_features, add_features_v2, add_features_v3


def main():
    FEATURE_VERSION = 3

    train, test = load_data()
    targets = build_targets(train)

    if FEATURE_VERSION == 1:
        train = add_features(train)
        test = add_features(test)
    elif FEATURE_VERSION == 2:
        train = add_features_v2(train)
        test = add_features_v2(test)
    elif FEATURE_VERSION == 3:
        train = add_features_v3(train)
        test = add_features_v3(test)
    else:
        raise ValueError(f"Unsupported FEATURE_VERSION: {FEATURE_VERSION}")

    model_list = ["lgbm"]

    pred_arrays = []

    for model_name in model_list:
        print(f"\n========== Running {model_name} ==========")

        if model_name in ["lgbm", "xgb", "rf"]:
            if FEATURE_VERSION == 1:
                feature_cols = get_GBM_feature_columns()
            elif FEATURE_VERSION == 2:
                feature_cols = get_GBM_feature_columns_v2()
            elif FEATURE_VERSION == 3:
                feature_cols = get_GBM_feature_columns_v3()
            else:
                raise ValueError(f"Unsupported FEATURE_VERSION: {FEATURE_VERSION}")

        elif model_name in ["logreg", "mlp"]:
            if FEATURE_VERSION == 1:
                feature_cols = get_NN_feature_columns()
            elif FEATURE_VERSION == 2:
                feature_cols = get_NN_feature_columns_v2()
            elif FEATURE_VERSION == 3:
                feature_cols = get_NN_feature_columns_v3()
            else:
                raise ValueError(f"Unsupported FEATURE_VERSION: {FEATURE_VERSION}")

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        X = train[feature_cols].copy()
        X_test = test[feature_cols].copy()

        test_preds, oof_scores = run_cv_training_dual(
            X, X_test, targets, model_name=model_name
        )

        pred_array = enforce_monotonicity(test_preds)

        save_submission(
            event_ids=test["event_id"],
            pred_array=pred_array,
            output_path=f"data/submissions/submission_{model_name}.csv",
        )

        pred_arrays.append(pred_array)

    final_pred = average_ensemble(pred_arrays)

    save_submission(
        event_ids=test["event_id"],
        pred_array=final_pred,
        output_path="data/submissions/submission_ensemble.csv",
    )


if __name__ == "__main__":
    main()