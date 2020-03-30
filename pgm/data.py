import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

aa_letters = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_ids = {l: i for i, l in enumerate(aa_letters)}

def np_onehot(a, shape):
    onehot = np.zeros(shape)
    onehot[np.arange(len(a)), a] = 1
    return onehot

class RawSequencesData(object):
    def __init__(self, file, size):
        df = pd.read_csv(file, index_col = 0)
        self.size = size if not (size%8) else (size - size%8 + 8)
        self.raw_sequences = list(df.seq.apply(lambda s: np.array([aa_ids[c] if c in aa_ids.keys() else -1 for c in s])))
        self.raw_sequences = [np_onehot(seq, (self.size, len(aa_ids))) for seq in self.raw_sequences]

    def __len__(self):
        return len(self.raw_sequences)
    
    def __getitem__(self, i):
        return self.raw_sequences[i], self.raw_sequences[i]
    
class AlignedSequencesData(object):
    def __init__(self, file):
        self.df = pd.read_csv(file, index_col = 0).drop_duplicates(["aligned_seq"])
        self.raw_sequences = np.array(list(self.df.aligned_seq.apply(lambda s: np.array([aa_ids[c] if c in aa_ids.keys() else -1 for c in s]))))
        self.raw_sequences = self.raw_sequences[:, ((self.raw_sequences == -1).max(0) == False)] 
        self.size = self.raw_sequences.shape[1]
        self.raw_sequences = [np_onehot(seq, (self.size, len(aa_ids))) for seq in self.raw_sequences]
        self.weights = self.df.weights.values
        
    def __len__(self):
        return len(self.raw_sequences)
    
    def __getitem__(self, i):
        return self.raw_sequences[i], self.weights[i]

class Seq_SS_Data(object):
    def __init__(self, file, size = 512):
        self.primary, self.ss3 = [], []
        df = pd.read_json(file)
        df["length"] = df.primary.apply(lambda x : len(x))
        df = df[df.length <= size]
        
        self.length = list(df.length)
        self.primary = list(df.primary.apply(lambda d : np_onehot(d, (size, 28))))
        self.ss3 = list(df.ss3.apply(lambda d : np_onehot(d, (size, 3))))
        del df

        self.primary = np.array(self.primary)
        self.primary = self.primary[:,:,np.where(np.sum(self.primary, axis = (0,1)) > 0)[0]]
        self.ss3 = np.array(self.ss3)
            
    def __len__(self):
        return len(self.primary)
    
    def __getitem__(self, i):
        return self.primary[i], self.ss3[i], self.length[i]