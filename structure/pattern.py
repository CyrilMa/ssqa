import numpy as np
import pickle

import torch
import torch.nn as nn
from .data import HMM_Data

inf = float("Inf")


class PatternMatching(nn.Module):
    def __init__(self, pattern, Q,
                 T=None, pi=None,
                 name="", MAX_SIZE=300):
        self.pattern = [pat[0] for pat in pattern]
        self.name = f"Matcher {name}"
        self.Q = torch.tensor(Q)
        self.T = torch.tensor(T)
        self.pi = torch.tensor(pi)

    def __repr__(self):
        return self.name

    def match(self, y):
        N = y.size(0)
        P = self.P_(y)
        Q = self.Q[:, :, :N + 1, :N + 1]
        P, Q = P.float(), Q.float()
        ll, a, b = self.LL(P, Q)
        s, boxes, _ = self.MLL(P, Q)
        return s, ll, a, b, P, Q

    def LL(self, P, Q):
        N = P.size(-1) - 1

        def sum_alpha():
            LL = []
            ll = -torch.ones(N + 1) * inf;
            ll[0] = 0
            LL.append(ll);
            last_ll = ll.view(-1, 1)
            for c in self.pattern:
                ll = torch.logsumexp(P[0, c] + Q[0, c] + last_ll, 0)
                last_ll = ll.view(-1, 1)
                LL.append(ll)
            return torch.cat([ll.view(1, -1) for ll in LL], 0)

        def sum_beta():
            LL = []
            ll = -torch.ones(N + 1) * inf;
            ll[-1] = 0
            LL.append(ll);
            last_ll = ll.view(1, -1)
            for c in self.pattern[::-1]:
                ll = torch.logsumexp(P[0, c] + Q[0, c] + last_ll, 1)
                last_ll = ll.view(1, -1)
                LL.append(ll)
            return torch.cat([ll.view(1, -1) for ll in LL[::-1]], 0)

        a, b = sum_alpha(), sum_beta()
        ll = a + b
        return ll, a, b

    def MLL(self, P, Q):
        N = P.size(-1) - 1

        def max_alpha():
            c_0 = self.pattern[0]
            LL = []
            last_ll = -torch.ones(N + 1) * inf;
            last_ll[0] = 0
            last_ll = last_ll.view(-1, 1)
            LL.append(last_ll)
            for c in self.pattern:
                ll, _ = (P[0, c] + Q[0, c] + last_ll).max(0)
                last_ll = ll.view(-1, 1)
                LL.append(ll)
            return [ll for ll in LL]

        def argmax_alpha(alpha):
            c_star, t_star = 3, N
            mll = []
            for a, c in zip(alpha, self.pattern[::-1]):
                t_star = torch.argmax(P[0, c_star, :, t_star] + Q[0, c_star, :, t_star] + a).item()
                c_star = c
                mll.append((c_star, t_star))
            return mll

        alpha = max_alpha()
        mll = argmax_alpha(alpha[::-1])[::-1]

        m, M = -1, 0
        s = torch.zeros(N).int()
        for c, t in mll:
            m, M, = M, t
            s[m:M] = int(c)
        return s, mll, alpha

    def P_(self, y):
        C = torch.log(y)
        N = C.shape[0]
        P = torch.zeros(4, N + 1, N + 1)
        for i in range(N + 1):
            P[:, i, :i + 1] = -inf
            if i == N:
                break
            P[0, :i + 1, i + 1:] += C[i, 0]
            P[1, :i + 1, i + 1:] += C[i, 1]
            P[2, :i + 1, i + 1:] += C[i, 2]

        P[3] = -inf
        for i in range(N + 1):
            P[3, i, i] = 0
        return P.view(1, *P.size())


def set_const(dataset, file, max_size=400):
    train_dataset = HMM_Data(f"{dataset}/{file}")
    bboxes = [[], [], []]
    T = np.zeros((4, 4))
    pi = np.zeros(4)
    for s in train_dataset.bbox:
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
    pickle.dump((Q, T, pi), open(f"{dataset}/statistics.pkl", "wb"))
    return Q, T, pi
