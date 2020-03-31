import numpy as np
import pickle

from .utils import to_onehot


class HMM_Data(object):
    def __init__(self, file):
        self.primary, self.ss3 = [], []
        data = pickle.load(open(file, 'rb'))
        self.primary = [p[:, :50] for p, _ in data.values()]
        self.target = [np.concatenate((p[:, 51:57],
                                       p[:, 65:],
                                       np.argmax(p[:, 57:65], 1).reshape(p.shape[0], 1),
                                       np.array(s).reshape(len(s), 1)), axis=1) for p, s in data.values()]
        ss3 = [to_onehot(s, (None, 3)) for _, s in data.values()]
        del data

        self.bbox = []
        for ss in ss3:
            ss = np.pad(ss, ((1, 1), (0, 0)), "constant")
            dss = (ss[1:] - ss[:-1])
            cls = to_onehot(np.where(dss == -1)[1], (None, 3)).T
            self.bbox.append(np.array([np.where(dss == 1)[0], np.where(dss == -1)[0], *cls]).T)

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, i):
        return self.primary[i], self.bbox[i], self.target[i]
