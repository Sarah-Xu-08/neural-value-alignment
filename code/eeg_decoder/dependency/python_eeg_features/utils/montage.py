from typing import Optional, List, Literal, Union
import numpy as np


_default_unipolar_montage = [
    "fp1", "fp2", "f7", "f8", 
    "t3", "t4", "t5", "t6", 
    "o1", "o2", "f3", "f4", 
    "c3", "c4", "p3", "p4", 
    "fz", "cz", "pz",
]

_default_bipolar_montage = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1", 
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",  
    "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

tuh_bipolar_montage = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1", 
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2", 
    "A1-T3", "T3-C3", "C3-CZ", "CZ-C4", 
    "C4-T4", "T4-A2", "FP1-F3", "F3-C3", 
    "C3-P3", "P3-O1", "FP2-F4", "F4-C4", 
    "C4-P4", "P4-O2",
]

_default_ref_electrode = ["a1", "a2"]

default_av_electrodes = [
    "fp1", "fp2", "f7", "f8", 
    "t3", "t4", "t5", "t6", 
    "o1", "o2", "f3", "f4", 
    "c3", "c4", "p3", "p4", 
    "fz", "cz", "pz",
]

default_av_electrodes_with_a1a2 = [
    "fp1", "fp2", "f7", "f8", 
    "t3", "t4", "t5", "t6", 
    "o1", "o2", "f3", "f4", 
    "c3", "c4", "p3", "p4", 
    "fz", "cz", "pz", "a1",
    "a2",
]

def _find_matched_channel(tar_ch: str, ch_names: List[str]) -> Optional[str]:
    for ch in ch_names:
        if tar_ch.lower() in ch.lower():
            return ch
    return None
       

def _find_left_or_right(ch: str) -> Literal['m', 'r', 'l']:
    ch_id = ''.join(filter(lambda c: '0' <= c <= '9', ch))
    if ch_id == "":
        return 'm'
    elif int(ch_id) % 2 == 0:
        return 'r'
    else:
        return 'l'


def rearrange_unipolar(
    a: np.ndarray,
    ch_names: List[str],
    montage: Optional[List] = None,
    ref_electrode: Optional[List[str]] = None,
    warn_missing: bool = True,
):
    ch2idx = {ch: i for i, ch in enumerate(ch_names)}
    if montage is None:
        montage = _default_unipolar_montage
        
    if ref_electrode is None:
        ref_electrode = _default_ref_electrode 

    found_ref = []
    for ref_ch in ref_electrode:
        matched = _find_matched_channel(ref_ch, ch_names)
        if matched:
            found_ref.append(matched)
        
    if len(found_ref) == 0:
        ref_left = ref_right = ref_mid = np.zeros_like(a[0])
    elif len(ref_electrode) == len(found_ref) == 2:
        pos0 = _find_left_or_right(ref_electrode[0])
        pos1 = _find_left_or_right(ref_electrode[1])
        assert (pos0 + pos1) in ['lr', 'rl'], "When 2 reference electordes are provided, there must be one on the left and the other on the right."
        if pos0 == "l":
            ref_left =  a[ch2idx[found_ref[0]]]
            ref_right = a[ch2idx[found_ref[1]]]
        else:
            ref_left =  a[ch2idx[found_ref[1]]]
            ref_right = a[ch2idx[found_ref[0]]]
        ref_mid = (ref_left + ref_right) / 2
    else:
        ref_left = ref_right = ref_mid = \
            np.mean(np.stack([a[ch2idx[ref]] for ref in found_ref]), 0)
        
    rearranged = []
    missing = []
    for ch in montage:
        pos = _find_left_or_right(ch)
        matched = _find_matched_channel(ch, ch_names)
        if not matched:
            missing.append(ch)
            rearranged.append(np.zeros_like(a[0]))
        elif pos == 'l':
            rearranged.append(a[ch2idx[matched]] - ref_left)
        elif pos == 'r':
            rearranged.append(a[ch2idx[matched]] - ref_right)
        else:
            rearranged.append(a[ch2idx[matched]] - ref_mid)
            
    if len(missing) > 0 and warn_missing:
        print(f"WARNING: found missing channels: {missing}")
        
    return np.stack(rearranged), missing


def rearrange_bipolar(
    a: np.ndarray,
    ch_names: List[str],
    montage: Optional[List] = None,
    warn_missing: bool = True,
):
    ch2idx = {ch: i for i, ch in enumerate(ch_names)}
    if montage is None:
        montage = _default_bipolar_montage
    
    rearranged = []
    missing = []
    for ch in montage:
        ch1, ch2 = ch.split("-")
        matched_1 = _find_matched_channel(ch1, ch_names)
        matched_2 = _find_matched_channel(ch2, ch_names)
        if not matched_1 or not matched_2:
            missing.append(ch)
            rearranged.append(np.zeros_like(a[0]))
        rearranged.append(a[ch2idx[matched_1]] - a[ch2idx[matched_2]])
            
    if len(missing) > 0 and warn_missing:
        print(f"WARNING: found missing channels: {missing}")
        
    return np.stack(rearranged), missing