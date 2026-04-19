import numpy as np


def average_ensemble(pred_arrays, weights=None):
    """
    pred_arrays: list of np.array, shape = (n_samples, 4)
    """

    if weights is None:
        weights = [1 / len(pred_arrays)] * len(pred_arrays)

    blended = np.zeros_like(pred_arrays[0])

    for pred, w in zip(pred_arrays, weights):
        blended += pred * w

    return blended