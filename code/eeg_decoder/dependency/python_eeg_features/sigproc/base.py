from typing import Optional, Literal, Callable, Iterable, Tuple
import numpy as np

from scipy.signal import sosfiltfilt, butter
from scipy.ndimage import convolve1d


def smooth(
    x: np.ndarray,
    window_len: int = 9,
    winfunc: Callable[[int], np.ndarray] = np.hamming,
) -> np.ndarray:
    """
    Smooth the EEG signal by applying a window function.
    
    Args:
    x (np.ndarray): _description_
    window_len (int, optional): _description_. Defaults to 11.
    winfunc (Callable[[int], np.ndarray], optional): _description_. Defaults to np.hamming.
    """
    if window_len < 3:
        return x
    s = x
    w = winfunc(window_len)
    y = convolve1d(s, w, axis=1, mode='mirror')
    return y
    

def filter(
    x: np.ndarray, 
    fs: int,
    type: Literal["lp", "hp", "bp", "bs"],
    freq1: int,
    freq2: Optional[int] = None,
    order: Optional[int] = None,
) -> np.ndarray:
    """Apply a butterworth filter to the EEG signal.

    Args:
        x (np.ndarray): _description_
        fs (int): _description_
        type (Literal[&quot;lp&quot;, &quot;hp&quot;, &quot;bp&quot;, &quot;bs&quot;]): _description_
        freq1 (int): _description_
        freq2 (Optional[int], optional): _description_. Defaults to None.
        order (Optional[int], optional): _description_. Defaults to None.
    """    
    if type == "bp" or "bs":
        assert freq2 is not None, "Two frequencies should be provided when you're applying a bandpass or bandstop filter."
        
    if order is None:
        order = 5
    if type == "lp":
        type = "lowpass"
        param = freq1
    elif type == "hp":
        type = "highpass"
        param = freq1
    elif type == "bp":
        type = "bandpass"
        param = [freq1, freq2]
    elif type == "bs":
        type = "bandstop"
        param = [freq1, freq2]
        
    coff_sos = butter(order, param, type, fs=fs, output="sos")
    y = sosfiltfilt(coff_sos, x)
    return y


def fbank(
    a: np.ndarray,
    sr: float = 500.,
    winlen: float = 2.5,
    winstep: float = 1.,
    nfilt: int = 35,
    nfft: Optional[int] = None,
    lf: float = 0.,
    hf: float = 70.,
    winfunc: Callable[[int], np.ndarray] = np.hamming,
    fixed_bank: bool = False,
) -> np.ndarray:
    """_summary_

    Args:
        a (np.ndarray): _description_
        sr (float, optional): _description_. Defaults to 500..
        winlen (float, optional): _description_. Defaults to 2.5.
        winstep (float, optional): _description_. Defaults to 1..
        nfilt (int, optional): _description_. Defaults to 35.
        nfft (Optional[int], optional): _description_. Defaults to None.
        lf (float, optional): _description_. Defaults to 0..
        hf (float, optional): _description_. Defaults to 70..
        winfunc (Callable[[int], np.ndarray], optional): _description_. Defaults to np.hamming.
    """    
    nfft = nfft or calc_nfft(np.ceil(sr * winlen))
    frames = frameEEG(a, winlen * sr, winstep * sr, winfunc)
    spec = pow_spec(frames, nfft)

    if fixed_bank:
        fbanks = get_fixed_banks(
            [
                [1, 4],
                [4, 8],
                [8, 14],
                [14, 31], 
                [31, 49],
            ],
            nfft,
            sr,
        )
    else:
        fbanks = get_filterbanks(nfilt, nfft, sr, lf, hf)

    feat = np.dot(spec, fbanks.T)
    
    return feat


def logfbank(
    a: np.ndarray,
    sr: float = 500.,
    winlen: float = 2.5,
    winstep: float = 1.,
    nfilt: int = 35,
    nfft: Optional[int] = None,
    lf: float = 0.,
    hf: float = 70.,
    winfunc: Callable[[int], np.ndarray] = np.hamming,
    fixed_bank: bool = False,
) -> np.ndarray:
    """_summary_

    Args:
        a (np.ndarray): _description_
        sr (float, optional): _description_. Defaults to 500..
        winlen (float, optional): _description_. Defaults to 2.5.
        winstep (float, optional): _description_. Defaults to 1..
        nfilt (int, optional): _description_. Defaults to 35.
        nfft (Optional[int], optional): _description_. Defaults to None.
        lf (float, optional): _description_. Defaults to 0..
        hf (float, optional): _description_. Defaults to 70..
        winfunc (Callable[[int], np.ndarray], optional): _description_. Defaults to np.hamming.
    """    
    return np.log2((fbank(a, sr, winlen, winstep, nfilt, nfft, lf, hf, winfunc, fixed_bank) + 1e-30) * 100)

    
def frameEEG(
    a: np.ndarray,
    f_len: float,
    f_step: float,
    winfunc: Callable[[int], np.ndarray] = np.hamming
):
    s_len = a.shape[-1]
    f_len = int(np.ceil(f_len))
    f_step = int(np.ceil(f_step))
    if s_len <= f_len:
        numframes = 1
    else:
        numframes = 1 + int(np.ceil((1.0 * s_len - f_len) / f_step))

    padlen = int((numframes - 1) * f_step + f_len)

    zeros = np.zeros((*(a.shape[:-1]), padlen - s_len,))
    padsignal = np.concatenate((a, zeros), -1)
    indices = np.tile(np.arange(0, f_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * f_step, f_step), (f_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[..., indices]
    win = np.tile(winfunc(f_len), (numframes, 1))
    return frames * win


def calc_nfft(point_num: int):
    nfft = 1
    while nfft < point_num:
        nfft *= 2
    return nfft

def mag_spec(a: np.ndarray, nfft: int):
    return np.abs(np.fft.rfft(a, nfft))

def pow_spec(a: np.ndarray, nfft: int):
    return 1.0 / nfft * np.square(mag_spec(a, nfft))

def hz2cub(hz: float):
    temp = 2793 + hz * (-450 + 25 * hz)
    temp = np.sqrt(temp)
    temp = (temp - 5 * hz + 45) ** (1/3)
    return -3 * ((3/2) ** (1/3)) * temp + 12 * (2 ** (1/3)) * (3 ** (2/3)) / temp + 9

def cub2hz(cub: float):
    c = 7 / 5
    b = -1 / 15
    a = 1 / 405
    return cub * (c + cub * (b + cub * a))

def get_filterbanks(
    nfilt: int = 35, 
    nfft: int = 1024,
    sr: float = 500.,
    lf: float = 0.,
    hf: float = 70.,
):
    """Compute a Cubic-filterbank. The filters contain several magical numbers. 
        TODO: test whether the magic numbers are the best.

    Args:
        nfilt (int, optional): _description_. Defaults to 35.
        nfft (int, optional): _description_. Defaults to 1024.
        sr (float, optional): _description_. Defaults to 500..
        lf (float, optional): _description_. Defaults to 0..
        hf (float, optional): _description_. Defaults to 0..
    """    
    lc = hz2cub(lf)
    hc = hz2cub(hf)
    cubpoints = np.linspace(lc, hc, nfilt+2)
    freqs = cub2hz(cubpoints)
    # print(freqs)
    
    bin = np.floor((nfft + 1) * cub2hz(cubpoints) / sr)
    
    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = 2 * (i - bin[j]) / (bin[j+1]-bin[j]) / (bin[j+2] - bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = 2 * (bin[j+2]-i) / (bin[j+2]-bin[j+1]) / (bin[j+2] - bin[j])
    return fbank


def get_fixed_banks(
    filt_bank: Iterable[Tuple[int, int]],
    nfft: int = 1024,
    sr: int = 200,
):
    fbank = np.zeros([len(filt_bank), nfft // 2 + 1])
    for i, bank in  enumerate(filt_bank):
        start = int((nfft + 1) * bank[0] / sr)
        end = int((nfft + 1) * bank[1] / sr)
        for j in range(start, end):
            fbank[i, j] = 1 / (end - start)
    return fbank


if __name__ == "__main__":
    # # test hz2cub
    # print("hz: ", 9, "cub: ", hz2cub(9))
    # print("hz: ", 0, "cub: ", hz2cub(0))
    # print("hz: ", 0.5, "cub: ", hz2cub(0.5))
    # print("hz: ", 20, "cub: ", hz2cub(20))
    # print("hz: ", 45, "cub: ", hz2cub(45))
    # print("hz: ", 70, "cub: ", hz2cub(70))
    
    # # test cub2hz
    # print("cub: ", 9, "hz: ", cub2hz(9))
    # print("cub: ", 0, "hz: ", cub2hz(0))
    # print("cub: ", 12, "hz: ", cub2hz(12))
    # print("cub: ", 20, "hz: ", cub2hz(20))
    # print("cub: ", 30, "hz: ", cub2hz(30))
    
    # # test the loop
    # print("before: ", 9, "after: ", cub2hz(hz2cub(9)))
    # print("before: ", 0, "after: ", cub2hz(hz2cub(0)))
    # print("before: ", 12, "after: ", cub2hz(hz2cub(12)))
    # print("before: ", 24, "after: ", cub2hz(hz2cub(24)))
    # print("before: ", 48, "after: ", cub2hz(hz2cub(48)))
    # print("before: ", 70, "after: ", cub2hz(hz2cub(70)))
    
    # compute filter banks
    winlen = 0.81
    winstep = 0.81
    nfilt= 10
    # nfilt = [[1,4],[4,8],[8,14],[14, 31], 
    #             [31, 49]]
    nfft = 1024
    lf = 0.
    hf = 70.
    sr = 1000

    get_filterbanks(nfilt, nfft, sr, lf, hf)



