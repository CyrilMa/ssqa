import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

inf = float("Inf")

class PatternMatching(nn.Module):

    def __init__(self, model_ss3, pattern, Q, seq_hmm, ss_hmm, size=68, name=""):
        super(PatternMatching, self).__init__()
        self.name = f"Matcher {name}"
        self.size = size
        self.pattern = [pat[0] for pat in pattern]
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.Q = Q.view(1, *Q.size())
        self.model_ss3 = model_ss3
        self.SEQ_HMM = seq_hmm
        self.SS_HMM = ss_hmm
        self.do_match = True

    def __repr__(self):
        return self.name

    def forward(self, x, single_marginalize = True, double_marginalize = True):
        batch_size = x.size(0)
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        ls = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x_[0] == 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[:20, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
        ls = torch.tensor(ls)

        hmm = hmm.to(self.model_ss3.device)
        y = F.softmax(self.model_ss3(hmm)[2].cpu(),1)
        P = self.P_(y).float()
        a = self.sum_alpha(P)
        b = self.sum_beta(P, ls)
        ll = a + b
        M = a[torch.arange(batch_size), -1, ls]
        p, t = None, None
        if single_marginalize:
            p = self.single_marginalize(P, a, b, M)
        if double_marginalize:
            t = self.double_marginalize(P, a, b, M)
        return M, ll, a, b, ls, p, t

    def single_marginalize(self, P, a, b, M):
        batch_size, len_pat, _ = a.size()
        p = a+b-M.view(batch_size, 1, 1)
        return p

    def double_marginalize(self, P, a, b, M):
        batch_size, len_pat, _ = a.size()
        t = a[:, :-1].view(batch_size, len_pat-1,-1,1)+b[:, 1:].view(batch_size, len_pat-1,1,-1)+P[:, 0, self.pattern]+\
            self.Q[:, 0, self.pattern]-M.view(batch_size, 1, 1, 1)
        return t

    def sum_alpha(self, P):
        batch_size = P.size(0)
        alpha, norm = [], []
        a_ = -torch.ones(batch_size, self.size+1) * inf
        a_[:,0] = 0
        n_ = a_.logsumexp(1).view(batch_size, 1)
        a_ -= n_
        alpha.append(a_), norm.append(n_)
        last_a, last_n = a_.view(batch_size, -1, 1), n_
        for c in self.pattern:
            a_ = torch.logsumexp(P[:, 0, c] + self.Q[:, 0, c] + last_a, 1)
            n_ = last_n + a_.logsumexp(1).view(batch_size, 1)
            a_ = a_ + last_n - n_
            last_a, last_n = a_.view(batch_size, -1, 1), n_
            alpha.append(a_), norm.append(n_)
        alpha = torch.cat([a_.view(batch_size, 1, -1) for a_ in alpha], 1)
        norm = torch.cat([n_.view(batch_size, 1, 1) for n_ in norm], 1)
        return alpha + norm

    def sum_beta(self, P, ls):
        batch_size = P.size(0)
        beta, norm = [], []
        b_ = -torch.ones(batch_size, self.size+1) * inf
        b_[torch.arange(batch_size),ls] = 0
        n_ = b_.logsumexp(1).view(batch_size, 1)
        b_ -= n_
        beta.append(b_), norm.append(n_)
        last_b, last_n = b_.view(batch_size, 1, -1), n_
        for c in self.pattern[::-1]:
            b_ = torch.logsumexp(P[:, 0, c] + self.Q[:, 0, c] + last_b, -1)
            n_ = last_n + b_.logsumexp(1).view(batch_size, 1)
            b_ = b_ + last_n - n_
            last_b, last_n = b_.view(batch_size, 1, -1), n_
            beta.append(b_), norm.append(n_)
        beta = torch.cat([b_.view(batch_size, 1, -1) for b_ in beta[::-1]], 1)
        norm = torch.cat([n_.view(batch_size, 1, 1) for n_ in norm[::-1]], 1)
        return beta + norm

    def P_(self, y):
        C = torch.log(y)
        batch_size = C.size(0)
        P = torch.zeros(batch_size, 3, self.size + 1, self.size + 1)
        for i in range(self.size):
            P[:, :, i, :i + 1] = -inf
            if i == self.size:
                break
            P[:, :, :i + 1, i + 1:] += C[:, :, i].view(batch_size, 3, 1, 1)
        return P.view(batch_size, 1, *P.size()[1:])

def set_const(dataset, max_size=400):
    bboxes = [[], [], []]
    T = np.zeros((4, 4))
    pi = np.zeros(4)
    for s in dataset.bbox:
        last_bb = None
        for bb in s:
            idx = np.argmax(bb[2:])
            bboxes[idx].append(int(bb[1] - bb[0]))
            if last_bb is not None:
                T[last_bb, idx] += 1
                T[idx, last_bb] += 1
            else:
                pi[idx] += 1
            last_bb = idx
        pi[last_bb] += 1
    T[:, 3] = pi
    T[3, 3] = 1
    for i in range(T.shape[0]):
        T[i] /= T[i].sum()

    E = np.zeros((3, 33))
    E[0, :33] = (np.bincount(bboxes[0]) / len(bboxes[0]))[:33]
    E[1, :21] = (np.bincount(bboxes[1]) / len(bboxes[1]))[:21]
    E[2, :5] = (np.bincount(bboxes[2]) / len(bboxes[1]))[:5]
    pi = T[:, 3]
    T, E, pi = np.log(T), np.log(E), np.log(pi)

    e = E.shape[-1]
    Q = np.ones((4, max_size, max_size)) * (-np.inf)
    for i in range(max_size - e):
        Q[:3, i, i:i + e] = E
        Q[3, i, i] = 0
    T = T.reshape(*T.shape, 1, 1)
    pi = pi.reshape(4, 1)
    Q = Q.reshape(1, *Q.shape)
    pickle.dump((Q, T, pi), open(f"statistics.pkl", "wb"))
    return Q, T, pi
