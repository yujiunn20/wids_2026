from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_model(model_name: str):
    if model_name == "lgbm":
        return LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=10,
            min_child_samples=8,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    elif model_name == "xgb":
        return XGBClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )

    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "logreg":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        return Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
            ))
        ])

    elif model_name == "mlp":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier

        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=32,
                learning_rate_init=1e-3,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            ))
        ])

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")