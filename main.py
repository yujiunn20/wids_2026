from src.data_loader import load_data
from src.features import get_GBM_feature_columns, get_NN_feature_columns
from src.targets import build_targets
from src.trainer import run_cv_training
from src.postprocess import enforce_monotonicity
from src.submission import save_submission
from src.ensemble import average_ensemble
from src.feature_engineering import add_features


def main():
    train, test = load_data()
    targets = build_targets(train)

    train = add_features(train)
    test = add_features(test)

    model_list = ["lgbm"]

    pred_arrays = []

    for model_name in model_list:
        print(f"\n========== Running {model_name} ==========")

        if model_name in ["lgbm", "xgb", "rf"]:
            feature_cols = get_GBM_feature_columns()
        elif model_name in ["logreg", "mlp"]:
            feature_cols = get_NN_feature_columns()
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        X = train[feature_cols].copy()
        X_test = test[feature_cols].copy()

        test_preds, oof_scores = run_cv_training(
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