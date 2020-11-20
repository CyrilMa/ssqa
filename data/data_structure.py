import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

import numpy as np
import pickle

from torch.utils.data import Dataset

class RBMSequenceStructureData(Dataset):

    def __init__(self, path, dataset="full"):
        super(RBMSequenceStructureData, self).__init__()
        self.raw_sequences, self.ss_sequences, self.ss_transitions, self.hmms = pickle.load(open(f"{path}/rbm_data.pkl", "rb"))
        self.weights = np.array(pickle.load(open(f"{path}/weights.pkl", "rb")))

        if dataset != "full":
            idx = np.array(pickle.load(open(f"{path}/is_val.pkl", "rb")))
            if dataset == "train":
                idx = 1 - idx
            idx = np.array(idx, dtype=bool)
            self.raw_sequences, self.ss_sequences, self.ss_transitions, self.hmms = \
                self.raw_sequences[idx], self.ss_sequences[idx], self.ss_transitions[idx], self.hmms[idx]
            self.weights = np.ones(*self.weights[idx].shape)

    def __len__(self):
        return len(self.raw_sequences)

    def __getitem__(self, i):
        return self.raw_sequences[i], self.ss_sequences[i], self.ss_transitions[i], self.hmms[i], self.weights[i]

class RBMSequenceData(Dataset):

    def __init__(self, path, dataset="full"):
        super(RBMSequenceData, self).__init__()
        self.raw_sequences, self.ss_sequences, self.ss_transitions, self.hmms = pickle.load(open(f"{path}/rbm_data.pkl", "rb"))
        self.weights = np.array(pickle.load(open(f"{path}/weights.pkl", "rb")))

        if dataset != "full":
            idx = np.array(pickle.load(open(f"{path}/is_val.pkl", "rb")))
            if dataset == "train":
                idx = 1 - idx
            idx = np.array(idx, dtype=bool)
            self.raw_sequences, self.hmms = \
                self.raw_sequences[idx], self.hmms[idx]
            self.weights = np.ones(*self.weights[idx].shape)

    def __len__(self):
        return len(self.raw_sequences)

    def __getitem__(self, i):
        return self.raw_sequences[i], [], [], self.hmms[i], self.weights[i]


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
        data = pickle.load(open(file, 'rb'))
        self.input_data = [p[:, :nfeats] for p, _ in data.values()]
        self.target = [np.concatenate((p[:, 51:57],
                                       p[:, 65:],
                                       np.argmax(p[:, 57:65], 1).reshape(p.shape[0], 1),
                                       np.array(s).reshape(len(s), 1)), axis=1) for p, s in data.values()]
        del data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, i: int):
        return self.input_data[i], self.target[i]


class SecondaryStructureRawDataset(Dataset):
    r"""
    Dataset structure without annotation for training.
    """

    def __init__(self, file: str, nfeats: int = 50):
        r"""

        Args:
            file (str): path to HMM file
            nfeats (int): number of feats to keep in the hmm profile
        """
        data = pickle.load(open(file, 'rb'))
        self.primary = [p[:, :nfeats] for p in data.values()]

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, i):
        return self.primary[i], None






def collate_sequences(data):
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
