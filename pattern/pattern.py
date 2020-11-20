import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

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

    def solve(self, m:Matching):
        m.len_pat = len(self.pattern)
        hmm, m.ls = self.hmm_(m)
        hmm = hmm.to(self.model_ss3.device)
        m.ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(),1).detach()+1e-3
        m.P = self.P_(m).float()
        m.a = self.max_alpha(m)
        m.sol = self.argmax_alpha(m)
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

    def max_alpha(self, m):
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
        alpha = m.a
        batch_size, P, ls = m.batch_size, m.P, m.ls
        t_star =  ls
        max_pattern = [t_star.view(-1, 1)]

        for c, a in zip(self.pattern[::-1],alpha[1:]):
            a_ = (P[torch.arange(batch_size), :, c, :, t_star] + self.Q.expand(batch_size, -1, -1, -1, -1)[
                                                                          torch.arange(batch_size), :, c, :,
                                                                          t_star] + a).argmax(-1)
            t_star = a_
            max_pattern.append(t_star.view(-1, 1))
        max_pattern = torch.cat([p[:, None] for p in max_pattern[::-1]], 1)
        return max_pattern


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

class PatternMatchingLoss(PatternMatching):

    def __init__(self, model_ss3, pattern, Q, seq_hmm, ss_hmm, size=68, name=""):
        super(PatternMatchingLoss, self).__init__(model_ss3, pattern, Q, seq_hmm, ss_hmm, size, name)

    def forward(self, m:Matching):
        hmm, m.ls = self.hmm_(m)
        hmm = hmm.to(self.model_ss3.device)
        m.ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(), 1)
        m.P = self.P_(m).float()
        m.a = self.sum_alpha(m)
        m.M = m.a[torch.arange(m.batch_size), -1, m.ls]
        return m

class LocalPatternMatchingLoss(PatternMatching):

    def __init__(self, model_ss3, pattern, Q, seq_hmm, ss_hmm, size=68, name=""):
        super(LocalPatternMatchingLoss, self).__init__(model_ss3, pattern, Q, seq_hmm, ss_hmm, size, name)

    def forward(self, m:Matching, n):
        hmm, m.ls = self.hmm_(m)
        m.len_pat = len(self.pattern)
        hmm = hmm.to(self.model_ss3.device)
        m.ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(), 1)
        m.P = self.P_(m).float()
        m.a = self.sum_alpha(m)
        m.b = self.sum_beta(m)
        m.M = m.a[torch.arange(m.batch_size), -1, m.ls]
        m.t = self.t_marginalize(m)
        m.L = self.l_marginalize(m,n)
        return m.L


class PatternMatchingSolver(PatternMatching):

    def forward(self, m:Matching, m_pat = None, M_pat = None):
        m.len_pat = len(self.pattern)
        hmm, m.ls = self.hmm_(m)
        hmm = hmm.to(self.model_ss3.device)
        m.ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(),1).detach()+1e-3
        m.P = self.P_(m).float()
        m.a = self.max_alpha(m)
        m.sol = self.argmax_alpha(m)
        return m


class PatternWithoutMatching(PatternMatching):

    def __init__(self, model_ss3, pattern, Q, seq_hmm, ss_hmm, size=68, name=""):
        super(PatternWithoutMatching, self).__init__(model_ss3, pattern, Q, seq_hmm, ss_hmm)
        self.name = f"Matcher {name}"
        self.size = size
        self.pattern = [pat[0] for pat in pattern]
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.Q = Q.view(1, *Q.size())
        self.model_ss3 = model_ss3
        self.SEQ_HMM = seq_hmm
        self.SS_HMM = (ss_hmm/ss_hmm.pow(2).sum(1, keepdim=True).pow(0.5))

    def forward(self, m:Matching, m_pat = None, M_pat = None):
        batch_size = m.batch_size
        m.len_pat = len(self.pattern)
        hmm, m.ls, idxs = self.hmm_(m)
        hmm = hmm.to(self.model_ss3.device)
        m.ss3 = ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(),1).detach()+1e-3
        ss3_u = torch.zeros(batch_size, *self.SS_HMM.size())
        for i, (ss3_,idx) in enumerate(zip(ss3, idxs)):
            ss3_ = ss3_[:, :len(idx)]
            ss3_u[:,:,idx] = ss3_/(ss3_.pow(2).sum(0, keepdim=True).pow(0.5))
        return (ss3_u*self.SS_HMM[None]).sum(1)

    def hmm_(self, m):
        batch_size, x = m.batch_size, m.x
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        hmm_ss = torch.zeros(batch_size, 3, self.size)
        ls = []
        idxs = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x_[0] == 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[:20, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
            hmm_ss[i,:, :n_idx] = self.SS_HMM[:, idx]
            idxs.append(idx)
        ls = torch.tensor(ls)
        return hmm, ls, idxs
