import torch
from torch.utils.data import Dataset
import pandas as pd
import dependency.python_eeg_features.sigproc as spc
import numpy as np
from label_utils import label_transfer
from sklearn.model_selection import StratifiedShuffleSplit

from utilsd.config import Registry

import os


class DATASET(metaclass=Registry, name="dataset"):
    pass


@DATASET.register_module("game1")
class NVAG1Dataset(Dataset):
    def __init__(
            self, 
            root_path: str,
            subject_id: str, 
            banknum: int = 5,
            fold: int = 0,
            split: str = "train",
            pe_type: str = "pe", # pe, rpe, spe, srpe3
            fixed_bank: bool = False,
    ):
        assert subject_id in [
             "03"
        ]
        
    
        data_path = os.path.join(root_path, f"A{subject_id}_epochs_1.pkl")
        data = pd.read_pickle(data_path)
        epoches, data, label = data

        self.pe_type = pe_type
        
        data = data[:, :64] - data[:, 64:]


        label = np.asarray(label_transfer(label, 1, None, pe_type))

        sss = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
        index = list(sss.split(data, label))[fold]
        
        if split == "train":
            idx = index[0]
        elif split == "test":
            idx = index[1]
        else:
            raise ValueError("split must be train or test")
        
        all_data = data[idx]
        all_label = label[idx]
 
        
        filtered = spc.filter(all_data, 1000, "bp", 0.5, 70, 5)
        feat = spc.logfbank(a=filtered, sr=1000, winlen=0.81, winstep=0.81, \
                                nfilt=banknum, fixed_bank=fixed_bank)
        
        self.data = (feat + 50) / 100.
        self.label = all_label
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), self.label[idx]
    
    @property
    def num_classes(self):
        return 3 if self.pe_type=='srpe3' else 2
        

@DATASET.register_module("game2")
class NVAG2Dataset(Dataset):
    def __init__(
        self, 
        root_path: str,
        subject_id: str,
        banknum: int = 5, 
        split: str = "train",
        fold: int = 0,
        so_type: str = "sub", # sub, obj
        pe_type: str = "pe", # pe, rpe, spe, srpe4
        fixed_bank: bool=False,
    ):
        assert subject_id in [
             "03"
        ]
        
        data_path = os.path.join(root_path, f"A{subject_id}_epochs_2.pkl")
        data_ = pd.read_pickle(data_path)
        _, raw_data, raw_label = data_
        
        data = []
        label = []
        
        self.so_type = so_type
        self.pe_type = pe_type
        
        for i in range(len(raw_data)):
            for j in range(len(raw_data[i])):
                data.extend(raw_data[i][j])
                label.extend(label_transfer(raw_label[i][j], 2, so_type, pe_type))
                
        data = np.array(data)[:, 0]
        label = np.array(label)
        data = data[:, :64] - data[:, 64:]

        
        index_splits = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        
        index = list(index_splits.split(data, label))[fold]
        if split == "train":
            idx = index[0]
        elif split == "test":
            idx = index[1]
        else:
            raise ValueError("split must be train or test")
            
        all_data = data[idx]
        all_label = label[idx]
       
        filtered = spc.filter(all_data, 1000, "bp", 0.5, 70, 5)
        feat = spc.logfbank(a=filtered, sr=1000, winlen=0.81, winstep=0.81, \
                                nfilt=banknum, fixed_bank=fixed_bank) # TODO: feat?
        
        self.data = (feat + 50) / 100.
        self.label = all_label
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), self.label[idx]
    
    @property
    def num_classes(self):
        return 4 if self.pe_type=='srpe4' else 2




if __name__ == "__main__":


    raw_dir = '/root/capsule/data/eeg'
    dataset = NVAG1Dataset(
        root_path=raw_dir,
        subject_id= "03",
        fold= 0,
        pe_type= "srpe3",
        loo= False,
        fixed_bank= True

    )
    
    print(list(dataset.label).count(1))
