import numpy as np
import pickle

import torch

class SecondaryStructureDataset(object):

    def __init__(self, file, target = True):
        self.primary, self.ss3 = [], []
        data = pickle.load(open(file, 'rb'))
        self.primary = [p[:, :50] for p, _ in data.values()]
        if target:
            self.target = [np.concatenate((p[:, 51:57],
                                           p[:, 65:],
                                           np.argmax(p[:, 57:65], 1).reshape(p.shape[0], 1),
                                           np.array(s).reshape(len(s), 1)), axis=1) for p, s in data.values()]
        else:
            self.target = [None for _ in range(len(data))]
        del data

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, i):
        return self.primary[i], self.target[i]


def collate_sequences(data):
    batch_size = len(data)
    lengths = []
    for x, t in data:
        lengths.append(len(x))

    max_length = max(lengths)
    primary = torch.zeros(batch_size, max_length, 50)
    target = torch.zeros(batch_size, max_length, 11)
    for i, ((x, t), l) in enumerate(zip(data, lengths)):
        primary[i, :l] = torch.tensor(x)
        if t is not None:
            target[i, :l] = torch.tensor(t)
    is_empty = (primary.max(-1).values != 0).int().view(*primary.size()[:-1], 1)
    return primary, target, is_empty
