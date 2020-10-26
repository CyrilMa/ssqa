import torch
import torch.nn.functional as F

from .pattern import PatternMatching, Matching

inf = float("Inf")


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
