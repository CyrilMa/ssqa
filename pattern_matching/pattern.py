import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import trace

inf = float("Inf")

class Matching(object):
    def __init__(self, x):
        super(Matching, self).__init__()
        self.x = x
        self.batch_size = x.size(0)
        self.len_pat = None
        self.ss3 = None
        self.P = None
        self.M = None
        self.a = None
        self.b = None
        self.ll = None
        self.ls = None
        self.p = None
        self.t = None
        self.L = None


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

    def forward(self, m:Matching, m_pat = None, M_pat = None):
        batch_size = m.batch_size
        m.len_pat = len(self.pattern)
        hmm, m.ls = self.hmm_(m)
        hmm = hmm.to(self.model_ss3.device)
        m.ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(),1).detach()+1e-3
        if m_pat is not None:
            m.ls = m.ls * 0 + (M_pat-m_pat)
            m.ss3[:, :, :M_pat-m_pat] = m.ss3[:,:,m_pat: M_pat]
            m.ss3[:, :, M_pat - m_pat:] = 1e-3
        m.P = self.P_(m).float()
        m.a = self.sum_alpha(m)
        m.b = self.sum_beta(m)
        m.ll = m.a + m.b
        m.M = m.a[torch.arange(batch_size), -1, m.ls]
        m.p = self.p_marginalize(m)
        m.t = self.t_marginalize(m)
        m.L = self.l_marginalize(m)
        return m

    def t_marginalize(self, m):
        batch_size, len_pat, P, a, b, M = m.batch_size, m.len_pat, m.P, m.a, m.b, m.M
        t = a[:, :-1].view(batch_size, len_pat,-1,1)+b[:, 1:].view(batch_size, len_pat,1,-1)+P[:, 0, self.pattern]+\
            self.Q[:, 0, self.pattern] - M.view(batch_size, 1, 1, 1)
        return t

    @staticmethod
    def p_marginalize(m):
        batch_size, a, b, M = m.batch_size, m.a, m.b, m.M
        p = a+b-M.view(batch_size, 1, 1)
        return p

    @staticmethod
    def l_marginalize(m, n = 30):
        batch_size, len_pat, t = m.batch_size, m.len_pat, m.t
        L = torch.zeros((batch_size, len_pat, n))
        for j in range(n-1):
            L[:, :, j] = trace(torch.exp(t), offset=j)
        L[:,:, -1] = 1-L.sum(-1)
        return L

    def sum_alpha(self, m):
        batch_size, P = m.batch_size, m.P
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

    def sum_beta(self, m):
        batch_size, P, ls = m.batch_size, m.P, m.ls
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

    def hmm_(self, m):
        batch_size, x = m.batch_size, m.x
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        ls = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x_[0] == 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[:20, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
        ls = torch.tensor(ls)
        return hmm, ls

    def P_(self, m):
        batch_size, C = m.batch_size, torch.log(m.ss3)
        P = torch.zeros(batch_size, 3, self.size + 1, self.size + 1)
        for i in range(self.size+1):
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

