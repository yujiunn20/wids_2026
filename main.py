from src.data_loader import load_data
from src.features import get_GBM_feature_columns, get_NN_feature_columns
from src.targets import build_targets
from src.trainer import run_cv_training
from src.postprocess import enforce_monotonicity
from src.submission import save_submission


def main():
    train, test = load_data()

    feature_cols = get_GBM_feature_columns()

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    targets = build_targets(train)

    test_preds, oof_scores = run_cv_training(X, X_test, targets)

    pred_array = enforce_monotonicity(test_preds)

    save_submission(
        event_ids=test["event_id"],
        pred_array=pred_array,
        output_path="data/submissions/submission_kfold.csv",
    )


if __name__ == "__main__":
    main()