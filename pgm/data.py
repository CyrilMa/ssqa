import numpy as np
import pickle

from torch.utils.data import Dataset


class SequenceStructureData(Dataset):

    def __init__(self, path, dataset="full"):
        super(SequenceStructureData, self).__init__()
        self.raw_sequences, self.ss_sequences, self.ss_transitions, self.hmms = pickle.load(open(f"{path}/rbm_data.pkl", "rb"))
        self.weights = np.array(pickle.load(open(f"{path}/weights.pkl", "rb")))

        if dataset != "full":
            idx = np.array(pickle.load(open(f"{path}/is_val.pkl", "rb")))
            if dataset == "train":
                idx = 1 - idx
            idx = np.array(idx, dtype=bool)
            self.raw_sequences, self.ss_sequences, self.ss_transitions, self.hmms = \
                self.raw_sequences[idx], self.ss_sequences[idx], self.ss_transitions[idx], self.hmms[idx]
            self.weights = self.weights[idx]

    def __len__(self):
        return len(self.raw_sequences)

    def __getitem__(self, i):
        return self.raw_sequences[i], self.ss_sequences[i], self.ss_transitions[i], self.hmms[i], self.weights[i]


class SequenceStructureSamplingData(Dataset):

    def __init__(self, path, train=True):
        super(SequenceStructureSamplingData, self).__init__()
        idx = np.array(pickle.load(open(f"{path}/is_val.pkl", "rb")))
        hmm = np.array(pickle.load(open(f"{path}/hmm.pkl", "rb")))
        if train:
            idx = 1 - idx
        idx = np.array(idx, dtype=bool)
        self.raw_sequences, self.ss_sequences, self.ss_transitions = pickle.load(open(f"{path}/rbm_data.pkl", "rb"))
        self.raw_sequences, self.ss_sequences, self.ss_transitions = \
            self.raw_sequences[idx], self.ss_sequences[idx], self.ss_transitions[idx]
        self.weights = np.array(pickle.load(open(f"{path}/weights.pkl", "rb")))
        self.weights = self.weights[idx]

    def __len__(self):
        return len(self.raw_sequences)

    def __getitem__(self, i):
        return self.raw_sequences[i], self.ss_sequences[i], self.ss_transitions[i], self.weights[i]
