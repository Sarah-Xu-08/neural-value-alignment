import numpy as np

def find_peaks(
    a: np.array,
    patience: int = 10,
    threshold: float = 2e-5,
):
    """_summary_
    
    Args:
        a (np.array): The input EEG signal, the last axis is assumed to be the temporal dimension.
        patience (int, optional): The range to decide whether a local maxima is viewded as a peak. For a point at time step *t*, points within [t-patience, t+patience] will be checked. Defaults to 10.
        threshold (float, optional): Only maxima bypassing the volumn threshold will be considered as peaks. Defaults to 2e-5.    

    Returns:
        _type_: _description_
    """
    assert 2 * patience + 1 < a.shape[-1], "The length of the signal should at least contain one possible peak within the patience range."
    total_points = a.shape[-1]
    padding = np.ones_like(a[..., :patience]) * threshold
    padded_signal = np.concatenate([padding, a, padding], -1)
    comp_range = np.stack([padded_signal[..., i:i + total_points] for i in range(2 * patience + 1)])
    max_target = np.max(comp_range, 0)
    return np.logical_and(max_target == a, a > threshold)
