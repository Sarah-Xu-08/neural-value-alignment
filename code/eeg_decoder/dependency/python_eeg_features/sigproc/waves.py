from typing import List
import numpy as np

from ..utils.montage import rearrange_unipolar, default_av_electrodes
from ..temporal.base import find_peaks
from .base import filter, frameEEG


def detect_alpha_wave(
    a: np.ndarray, 
    ch_names: List[str],
    fs: float,
):
    a_o, missing = rearrange_unipolar(a, ch_names, ["o1", "o2"], default_av_electrodes)
    assert len(missing) == 0, "can't find electrode o1 and o2 for alpha wave detection"
    a_o = filter(a_o, fs, "bp", 0.5, 75)
    seg_len = seg_stride = 1.
    frames = frameEEG(a_o, fs * seg_len, fs * seg_stride) # 2 * frames * (fs * t)

def find_peaks(
    
):
    pass