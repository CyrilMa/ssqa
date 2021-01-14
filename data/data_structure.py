import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

class SSQAData(Dataset):
    def __init__(self, file, subset=None):
        super(SSQAData, self).__init__()
        data = torch.load(file)
        keys = list(data.keys())
        print("Available : ", *keys)
        if subset is not None:
            idx = data["subset"][subset]
        else:
            idx = torch.arange(data["L"])
        self.L = len(idx)

        self.seqs = data["seq"][idx] if "seq" in keys else None
        self.ss3 = data["ss3"][idx] if "ss3" in keys else None
        self.ss8 = data["ss8"][idx] if "ss8" in keys else None
        self.others = data["others"] if "others" in keys else None
        self.weights = data["weights"][idx] if "weights" in keys else None

        self.seq_hmm = data["seq_hmm"] if "seq_hmm" in keys else None
        self.ss_hmm = data["ss_hmm"] if "ss_hmm" in keys else None

        self.c_pattern3, self.n_pattern3, self.c_pattern8, self.n_pattern8 = data["pattern"] if "pattern" in keys else (None,None,None,None)

    def __len__(self):
        return self.L

    def __getitem__(self, i):
        pass

class SSQAData_SSinf(SSQAData):
    def __getitem__(self, i):
        idx = torch.where(self.seqs[i].sum(0)>0)[0]
        return torch.cat([self.seqs[i][:,idx], self.seq_hmm[:,idx]], 0)

class SSQAData_QA(SSQAData):
    def __getitem__(self, i):
        return torch.cat([self.seqs[i], self.seq_hmm], 0)

class SSQAData_RBM(SSQAData):
    def __getitem__(self, i):
        gaps = (self.seqs[i].sum(0) == 0).int()
        return torch.cat([gaps[None], self.seqs[i]], 0), self.weights[i]

class SecondaryStructureAnnotatedDataset(Dataset):
    r"""
    Dataset structure with annotation for training.
    """

    def __init__(self, file: str, nfeats: int = 50):
        r"""

        Args:
            file (str): path to HMM file
            nfeats (int): number of feats to keep in the hmm profile
        """
        data = torch.load(file)
        self.input_data = [x[:, :nfeats] for x in data]
        ss8 = [x[:, 57:65].argmax(1) for x in data]
        ss3 = [(2*((x==5) | (x==6) | (x==7)).int() + ((x==3) | (x==4)).int()) for x in ss8]
        self.target = [torch.cat([x[:, 51:57],x[:, 65:],s8[:,None],s3[:,None]],1) for (x,s8,s3) in zip(data,ss8, ss3)]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, i: int):
        return self.input_data[i], self.target[i]

def collate_sequences(data):
    r"""
    Append sequences of different size together.
    Args:
        data (list of tuples): iterator of sequences and targetted values (None if prediction task)
    """
    batch_size = len(data)
    feats = data[0].shape[-1]
    lengths = []
    for x in data:
        lengths.append(len(x))

    max_length = max(lengths)
    primary = torch.zeros(batch_size, max_length, feats)
    for i, (x, l) in enumerate(zip(data, lengths)):
        primary[i, :l] = torch.tensor(x)
    is_empty = (primary.max(-1).values != 0).int().view(*primary.size()[:-1], 1)
    return primary, is_empty


def collate_sequences_train(data):
    r"""
    Append sequences of different size together.
    Args:
        data (list of tuples): iterator of sequences and targetted values (None if prediction task)
    """
    batch_size = len(data)
    _, feats = data[0][0].shape
    lengths = []
    for x, t in data:
        lengths.append(len(x))

    max_length = max(lengths)
    primary = torch.zeros(batch_size, max_length, feats)
    target = torch.zeros(batch_size, max_length, 11)
    for i, ((x, t), l) in enumerate(zip(data, lengths)):
        primary[i, :l] = torch.tensor(x)
        if t is not None:
            target[i, :l] = torch.tensor(t)
    is_empty = (primary.max(-1).values != 0).int().view(*primary.size()[:-1], 1)
    return primary, target, is_empty
