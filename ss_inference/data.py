import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

class SecondaryStructureAnnotatedDataset(Dataset):

    def __init__(self, file, feats = 50):
        self.primary, self.ss3 = [], []
        data = pickle.load(open(file, 'rb'))
        self.primary = [p[:, :feats] for p, _ in data.values()]
        self.target = [np.concatenate((p[:, 51:57],
                                       p[:, 65:],
                                       np.argmax(p[:, 57:65], 1).reshape(p.shape[0], 1),
                                       np.array(s).reshape(len(s), 1)), axis=1) for p, s in data.values()]
        del data

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, i):
        return self.primary[i], self.target[i]
    
class SecondaryStructureRawDataset(Dataset):

    def __init__(self, file, feats = 50):
        self.primary, self.ss3 = [], []
        data = pickle.load(open(file, 'rb'))
        self.primary = [p[:, :feats] for p in data.values()]
        del data

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, i):
        return self.primary[i], None


def collate_sequences(data):
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
