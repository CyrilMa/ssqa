import numpy as np
import pickle


class SequenceStructureData(object):
    def __init__(self, filename):
        self.raw_sequences, self.ss_sequences, self.ss_transitions = pickle.load(open(filename, "rb"))
        self.weights = np.ones(len(self.raw_sequences))

    def __len__(self):
        return len(self.raw_sequences)

    def __getitem__(self, i):
        return self.raw_sequences[i], self.ss_sequences[i], self.ss_transitions[i], self.weights[i]
