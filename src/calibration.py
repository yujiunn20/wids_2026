from sklearn.isotonic import IsotonicRegression


def fit_isotonic_calibrator(y_true, y_pred):
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred, y_true)
    return calibrator


def apply_calibrator(calibrator, preds):
    return calibrator.transform(preds)