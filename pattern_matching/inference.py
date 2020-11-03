
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pattern import Matching

inf = float("Inf")


class PatternMatchingInference(nn.Module):

    def __init__(self, model_ss3, Q, pi, seq_hmm, size=68, name=""):
        super(PatternMatchingInference, self).__init__()
        self.name = f"Matcher {name}"
        self.size = size
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.Q = Q.view(1, *Q.size())
        self.pi = pi.view(1, *pi.size())
        self.model_ss3 = model_ss3
        self.SEQ_HMM = seq_hmm
        self.do_match = True

    def __repr__(self):
        return self.name

    def forward(self, m: Matching, T):
        hmm, m.ls = self.hmm_(m)
        hmm = hmm.to(self.model_ss3.device)
        m.ss3 = F.softmax(self.model_ss3(hmm)[2].cpu(), 1).detach() + 1e-3
        m.P = self.P_(m).float()
        m.a = self.max_alpha(m, T)
        return self.argmax_alpha(m, T)

    def max_alpha(self, m, T):
        batch_size, P, ls = m.batch_size, m.P, m.ls
        alpha, norm = [], []
        a_ = -torch.ones(batch_size, 4, self.size + 1) * inf
        a_[:, :, 0] = self.pi
        n_ = a_.logsumexp((-1, -2), keepdim=True)
        a_ -= n_
        alpha.append(a_), norm.append(n_)
        last_a, last_n = a_[:, :, None, :, None], n_
        for i in range(self.size):
            a_ = (P + T * self.Q + last_a).max(-4)[0].max(-2)[0]
            n_ = last_n + a_.logsumexp((-1, -2), keepdim=True)
            a_ = a_ + last_n - n_
            alpha.append(a_), norm.append(n_)
            last_a, last_n = a_[:, :, None, :, None], n_
        alpha = torch.cat([a_[:, None] for a_ in alpha[::-1]], 1)
        norm = torch.cat([n_[:, None] for n_ in norm], 1)
        return alpha + norm

    def argmax_alpha(self, m, T):
        alpha = m.a
        size = alpha.size(-1)
        batch_size, P, ls = m.batch_size, m.P, m.ls
        c_star, t_star = torch.ones(batch_size).long() * 3, ls
        max_pattern = [torch.cat([c_star.view(-1, 1), t_star.view(-1, 1)], 1)]

        for a in alpha.transpose(1, 0)[1:]:
            a_ = (P[torch.arange(batch_size), :, c_star, :, t_star] + T * self.Q.expand(batch_size, -1, -1, -1, -1)[
                                                                          torch.arange(batch_size), :, c_star, :,
                                                                          t_star] + a).view(batch_size, -1).argmax(-1)
            c_star, t_star = a_ // size, a_ % size
            max_pattern.append(torch.cat([c_star.view(-1, 1), t_star.view(-1, 1)], 1))
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
        batch_size, C, ls = m.batch_size, torch.log(m.ss3), m.ls
        P = torch.zeros(batch_size, 4, self.size + 1, self.size + 1)
        for i in range(self.size + 1):
            P[:, :3, i, :i + 1] = -inf
            if i == self.size:
                break
            P[:, :3, :i + 1, i + 1:] += C[:, :, i].view(batch_size, 3, 1, 1)
        P[:, 3] = -inf
        P[:, 3, ls, ls] = 0
        return P[:, None, :, :, :]
