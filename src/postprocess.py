import numpy as np

# 讓出來的預測機率符合單調遞增的特性
def enforce_monotonicity(test_preds):
    pred_array = np.column_stack([
        test_preds["prob_12h"],
        test_preds["prob_24h"],
        test_preds["prob_48h"],
        test_preds["prob_72h"],
    ])

    pred_array = np.maximum.accumulate(pred_array, axis=1)
    return pred_array