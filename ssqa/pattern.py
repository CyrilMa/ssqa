import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import trace

inf = float("Inf")

class Matching(object):
    def __init__(self, x):
        super(Matching, self).__init__()
        self.x = x
        self.batch_size = x.size(0)
        self.len_pat = None
        self.ss = None
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
    r"""
    Class to perform patter marching operations
    Args:
        model_ss (nn.Module): model to predict secondary structure
        pattern (list): pattern of the form [(r,m,M) ...]
        Q (torch.Tensor): Emission matrix
        seq_hmm (torch.tensor): HMM profile of the sequence
        size (int): max length of the sequence
        name (str): name of the model
    """

    def __init__(self, model_ss, pattern, Q, seq_hmm, size=68, name=""):
        super(PatternMatching, self).__init__()
        self.name = f"Matcher {name}"
        self.size = size
        self.ssn = Q.size(1)
        self.pattern = [pat[0] for pat in pattern]
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.Q = Q.view(1, *Q.size())
        self.model_ss = model_ss
        self.SEQ_HMM = seq_hmm

    def __repr__(self):
        return self.name

    def forward(self, m:Matching):
        r"""
        Perform Pattern Matching on m
        Args:
            m (Matching) : Data to match
        """
        batch_size = m.batch_size
        m.len_pat = len(self.pattern)
        hmm, m.ls = self.hmm_(m)
        hmm = hmm.to(self.model_ss.device)
        if self.ssn == 8:
            m.ss = F.softmax(self.model_ss(hmm)[1].cpu(),1).detach()+1e-3
        if self.ssn == 3:
            m.ss = F.softmax(self.model_ss(hmm)[2].cpu(),1).detach()+1e-3
        m.P = self.P_(m).float()
        m.a = self.sum_alpha(m)
        m.b = self.sum_beta(m)
        m.ll = m.a + m.b
        m.M = m.a[torch.arange(batch_size), -1, m.ls]
        m.t = self.t_marginalize(m)
        m.p = self.p_marginalize(m)
        m.L = self.l_marginalize(m)
        return m

    def t_marginalize(self, m):
        r"""
        Build t_k features for all k
        Args:
            m (Matching) : Data to match

        """
        P, a, b, M, Q = m.P, m.a, m.b, m.M, self.Q
        t = a[:, :-1, :, None]+b[:, 1:, None, :] + P[:, 0, self.pattern]+ Q[:, 0, self.pattern] - M[:,None,None,None]
        return t

    def p_marginalize(self, m):
        r"""
        Build p_k features for all k (unused)
        Args:
            m (Matching) : Data to match
        """
        P, Q, t = m.P, self.Q, m.t
        p = []
        for i, c in enumerate(self.pattern):
            p_ = (t[:,i]+Q[:,0,c]+P[:,0,c]).logsumexp((-1,-2))
            p.append(p_)
        p = torch.cat([x[:,None] for x in p], 1)
        return p

    @staticmethod
    def l_marginalize(m, n = 30):
        r"""
        Build l_k features for all k (unused)
        Args:
            m (Matching) : Data to match
            n (int) : maximal length to marginalize
        """

        batch_size, len_pat, t = m.batch_size, m.len_pat, m.t
        L = torch.zeros((batch_size, len_pat, n))
        for j in range(n-1):
            L[:, :, j] = trace(torch.exp(t), offset=j)
        L[:,:, -1] = 1-L.sum(-1)
        return L

    def sum_alpha(self, m):
        r"""
        Compute alpha_k for maginalizing
        Args:
            m (Matching) : Data to match
        """

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
        r"""
        Compute beta_k for maginalizing
        Args:
            m (Matching) : Data to match
        """

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

    def max_alpha(self, m):
        r"""
        Compute max alpha for solving
        Args:
            m (Matching) : Data to match
        """

        batch_size, P = m.batch_size, m.P
        alpha, norm = [], []
        a_ = -torch.ones(batch_size, self.size+1) * inf
        a_[:,0] = 0
        n_ = a_.logsumexp(1).view(batch_size, 1)
        a_ -= n_
        alpha.append(a_), norm.append(n_)
        last_a, last_n = a_.view(batch_size, -1, 1), n_
        for c in self.pattern:
            a_ = (P[:, 0, c] + self.Q[:, 0, c] + last_a).max(1)[0]
            n_ = last_n + a_.logsumexp(1).view(batch_size, 1)
            a_ = a_ + last_n - n_
            last_a, last_n = a_.view(batch_size, -1, 1), n_
            alpha.append(a_), norm.append(n_)
        alpha = torch.cat([a_.view(batch_size, 1, -1) for a_ in alpha[::-1]], 1)
        norm = torch.cat([n_.view(batch_size, 1, 1) for n_ in norm[::-1]], 1)
        return alpha + norm

    def argmax_alpha(self, m):
        r"""
        Compute argmax alpha for solving
        Args:
            m (Matching) : Data to match
        """

        alpha, batch_size = m.a.permute(1,0,2).detach(), m.batch_size
        P, ls, Q = m.P, m.ls, self.Q.expand(batch_size, -1, -1, -1, -1)
        t_star =  ls
        max_pattern = [t_star[:,None]]

        for c, a in zip(self.pattern[::-1],alpha[1:]):
            a_ = (P[torch.arange(batch_size), :, c, :, t_star] + Q[torch.arange(batch_size), :, c, :, t_star] + a).argmax(-1)
            t_star = a_
            max_pattern.append(t_star[:,None])
        max_pattern = torch.cat([p[:,None] for p in max_pattern[::-1]], 1)
        return max_pattern


    def hmm_(self, m):
        r"""
        Compute hmm vector for pattern matching inference
        Args:
            m (Matching) : Data to match
        """

        batch_size, x = m.batch_size, m.x
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        ls = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x_[:20].sum(0) > 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[:20, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
        ls = torch.tensor(ls)
        return hmm, ls

    def P_(self, m):
        r"""
        Compute P for emission probability

        Args:
            m (Matching) : Data to match
        """

        batch_size, C = m.batch_size, torch.log(m.ss)
        P = torch.zeros(batch_size, self.ssn, self.size+1, self.size+1)
        for i in range(self.size+1):
            P[:, :, i:, :i+1] = -inf
            if i == self.size:
                break
            P[:, :, :i+1, i+1:] += C[:, :, i, None, None]
        return P[:,None]


