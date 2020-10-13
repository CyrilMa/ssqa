import torch
import torch.nn as nn
import torch.nn.functional as F

inf = float("Inf")

class PatternMatchingLoss(nn.Module):

    def __init__(self, model_ss3, pattern, Q, seq_hmm, ss_hmm, size=68):
        super(PatternMatchingLoss, self).__init__()
        self.size = size
        self.pattern = [pat[0] for pat in pattern]
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.Q = torch.log(torch.exp(Q) + 1e-8).view(1, *Q.size())
        self.model_ss3 = model_ss3
        self.SEQ_HMM = seq_hmm
        self.SS_HMM = ss_hmm
        self.do_match = True

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 21, -1)
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        ls = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x[i].argmax(0) != 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[1:, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
        ls = torch.tensor(ls)
        hmm = hmm.to(self.model_ss3.device)
        p_ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(),1)
        a = self.match(p_ss3)
        res = a[torch.arange(batch_size), ls] / ls
        return res

    def match(self, y):
        P = self.P_(y).float()
        P = torch.log(torch.exp(P)+1e-8)
        a = self.sum_alpha(P, self.Q)
        return a

    def sum_alpha(self, P, Q):
        batch_size = P.size(0)
        a_ = -torch.ones(batch_size, self.size + 1) * inf
        a_[:, 0] = 0
        n_ = a_.logsumexp(1).view(batch_size, 1)
        a_ -= n_
        last_a = a_.view(batch_size, -1, 1)
        last_n = n_
        for c in self.pattern:
            a_ = torch.logsumexp(P[:, 0, c] + Q[:, 0, c] + last_a, 1)
            n_ = last_n + a_.logsumexp(1).view(batch_size, 1)
            a_ = a_ + last_n - n_
            last_a = a_.view(batch_size, -1, 1)
            last_n = n_
        return a_ + n_

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

class QuickPatternMatchingLoss(nn.Module):

    def __init__(self, model_ss3, seq_hmm, ss_hmm, size=68):
        super(QuickPatternMatchingLoss, self).__init__()
        self.size = size
        self.model_ss3 = model_ss3
        self.SEQ_HMM = seq_hmm
        self.SS_HMM = ss_hmm
        self.do_match = True

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 21, -1)
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        weights = torch.zeros(batch_size, 3, self.size)
        ls = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x[i].argmax(0) != 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[1:, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
            weights[i,:,:n_idx] = self.SS_HMM[:, idx]
        ls = torch.tensor(ls)
        hmm = hmm.to(self.model_ss3.device)
        p_ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(),1)
        a = torch.log((weights * p_ss3).view(batch_size, -1).sum(-1) / ls)
        return a

class FlexiblePatternMatchingLoss(nn.Module):

    def __init__(self, model_ss3, Q, seq_hmm, ss_hmm, size=68):
        super(FlexiblePatternMatchingLoss, self).__init__()
        self.size = size
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.Q = torch.log(torch.exp(Q) + 1e-8).view(1, *Q.size())
        self.model_ss3 = model_ss3
        self.SEQ_HMM = seq_hmm
        self.SS_HMM = ss_hmm
        self.do_match = True

    def build_pattern(self, idxs):
        patterns = []
        for idx in idxs:
            ss = self.SS_HMM[idx].argmax(0)

        return

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 21, -1)
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        ls = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x[i].argmax(0) != 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[1:, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
        ls = torch.tensor(ls)
        hmm = hmm.to(self.model_ss3.device)
        p_ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(),1)
        a = self.match(p_ss3)
        res = a[torch.arange(batch_size), ls] / ls
        return res

    def match(self, y):
        P = self.P_(y).float()
        P = torch.log(torch.exp(P)+1e-8)
        a = self.sum_alpha(P, self.Q)
        return a

    def sum_alpha(self, P, Q):
        batch_size = P.size(0)
        a_ = -torch.ones(batch_size, self.size + 1) * inf
        a_[:, 0] = 0
        n_ = a_.logsumexp(1).view(batch_size, 1)
        a_ -= n_
        last_a = a_.view(batch_size, -1, 1)
        last_n = n_
        for c in self.pattern:
            a_ = torch.logsumexp(P[:, 0, c] + Q[:, 0, c] + last_a, 1)
            n_ = last_n + a_.logsumexp(1).view(batch_size, 1)
            a_ = a_ + last_n - n_
            last_a = a_.view(batch_size, -1, 1)
            last_n = n_
        return a_ + n_

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
