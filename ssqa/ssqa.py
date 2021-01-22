import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .pattern import *
from .dotproduct import *

inf = float("Inf")


class SSQA(nn.Module):
    r"""
        Class to perform SSQA
    Args:
        model_ss (nn.Module): model to predict secondary structure
        pattern (list): pattern of the form [(r,m,M) ...]
        Q (torch.Tensor): Emission matrix
        seq_hmm (torch.tensor): HMM profile of the sequence
        ss_hmm3 (torch.tensor): profile of the 3-class secondary structure
        ss_hmm8 (torch.tensor): profile of the 8-class secondary structure
        size (int): max length of the sequence
        name (str): name of the model

    """

    def __init__(self, model_ss, pattern=None, seq_hmm=None, ss_hmm3=None, ss_hmm8=None, name=""):
        super(SSQA, self).__init__()
        self.name = f"SSQA {name}"
        if seq_hmm is None:
            return
        self.seq_hmm, self.ss_hmm3, self.ss_hmm8 = seq_hmm, ss_hmm3, ss_hmm8
        self.size = seq_hmm.size(-1)
        self.c3, self.n3, self.c8, self.n8 = pattern
        self.model_ss = model_ss

        reg3 = [(i, None, None) for i in self.n3]
        Q3 = torch.ones(3, self.size + 1, self.size + 1) * (-inf)
        for i in range(self.size + 1):
            Q3[:3, i, i + 1:] = 0
        Q3 = Q3.view(1, 3, self.size + 1, self.size + 1)

        self.matcher3 = PatternMatching(model_ss, Q=Q3, seq_hmm=self.seq_hmm, pattern=reg3, size=self.size)
        self.dotproduct3 = DotProduct(model_ss, Q=Q3, seq_hmm=self.seq_hmm, ss_hmm=self.ss_hmm3, size=self.size)

        reg8 = [(i, None, None) for i in self.n8]
        Q8 = torch.ones(8, self.size + 1, self.size + 1) * (-inf)
        for i in range(self.size + 1):
            Q8[:8, i, i + 1:] = 0
        Q8 = Q8.view(1, 8, self.size + 1, self.size + 1)

        self.matcher8 = PatternMatching(model_ss, Q=Q8, seq_hmm=self.seq_hmm, pattern=reg8, size=self.size)
        self.dotproduct8 = DotProduct(model_ss, Q=Q8, seq_hmm=self.seq_hmm, ss_hmm=self.ss_hmm8, size=self.size)

        self.dpmin, self.dpmax = None, None
        self.pmmin, self.pmmax = None, None
        self.dpclf = RandomForestClassifier(200)
        self.pmclf = RandomForestClassifier(200)

        self.trained_unsupervised = False
        self.trained_supervised = False

    def save(self, path):
        state = ((self.seq_hmm, self.ss_hmm3, self.ss_hmm8, self.size, self.c3, self.n3, self.c8,
                  self.n8, self.trained_unsupervised, self.trained_supervised, self.pm0),
                 self.dotproduct3.state_dict(), self.dotproduct8.state_dict(),
                 self.matcher3.state_dict(), self.matcher8.state_dict(),
                 self.dpclf, self.dpmin, self.dpmax,
                 self.pmclf, self.pmmin, self.pmmax)
        torch.save(state, path)

    def load(self, path):
        state, dp3_state, dp8_state, pm3_state, pm8_state, self.dpclf, self.dpmin, self.dpmax, self.pmclf, self.pmmin, self.pmmax = torch.load(
            path)
        self.seq_hmm, self.ss_hmm3, self.ss_hmm8, self.size, self.c3, self.n3, self.c8, self.n8, self.trained_unsupervised, self.trained_supervised, self.pm0 = state
        reg3 = [(i, None, None) for i in self.n3]
        Q3 = torch.ones(3, self.size + 1, self.size + 1) * (-inf)
        for i in range(self.size + 1):
            Q3[:3, i, i + 1:] = 0
        Q3 = Q3.view(1, 3, self.size + 1, self.size + 1)

        self.matcher3 = PatternMatching(self.model_ss, Q=Q3, seq_hmm=self.seq_hmm, pattern=reg3, size=self.size)
        self.dotproduct3 = DotProduct(self.model_ss, Q=Q3, seq_hmm=self.seq_hmm, ss_hmm=self.ss_hmm3, size=self.size)

        reg8 = [(i, None, None) for i in self.n8]
        Q8 = torch.ones(8, self.size + 1, self.size + 1) * (-inf)
        for i in range(self.size + 1):
            Q8[:8, i, i + 1:] = 0
        Q8 = Q8.view(1, 8, self.size + 1, self.size + 1)

        self.matcher8 = PatternMatching(self.model_ss, Q=Q8, seq_hmm=self.seq_hmm, pattern=reg8, size=self.size)
        self.dotproduct8 = DotProduct(self.model_ss, Q=Q8, seq_hmm=self.seq_hmm, ss_hmm=self.ss_hmm8, size=self.size)

        self.dotproduct3.load_state_dict(dp3_state)
        self.dotproduct8.load_state_dict(dp8_state)
        self.matcher3.load_state_dict(pm3_state)
        self.matcher8.load_state_dict(pm8_state)

    def forward(self, x):
        m3 = Matching(x)
        m8 = Matching(x)
        self.matcher3(m3)
        self.dotproduct3(m3)
        self.matcher8(m8)
        self.dotproduct8(m8)
        return m3, m8

    def normalize(self, dp, pm):
        dp = ((dp - self.dpmin) / (self.dpmax - self.dpmin).clamp(1e-8, 1)).clamp(0, 1)
        pm = ((pm - self.pmmin) / (self.pmmax - self.pmmin).clamp(1e-8, 1)).clamp(0, 1)
        return dp, pm

    def featuring(self, X, reference_idx=[0]):
        dp3 = []
        pm3 = []
        dp8 = []
        pm8 = []
        batch_size = 32
        N = len(X)
        for batch_idx in range(N // batch_size + 1):
            x = X[batch_idx * batch_size: (batch_idx + 1) * batch_size].float()
            m3, m8 = self(x)
            dp3.append(m3.dp.detach())
            dp8.append(m8.dp.detach())
            pm3.append(m3.L.detach())
            pm8.append(m8.L.detach())
            torch.cuda.empty_cache()
        dp3 = torch.cat(dp3, 0)
        dp8 = torch.cat(dp8, 0)
        pm3 = torch.cat(pm3, 0)
        pm8 = torch.cat(pm8, 0)

        u = torch.arange(30)[None, None]
        pm = torch.cat([pm3, pm8], 1).clamp(1e-8, 1)
        if reference_idx is not None:
            self.pm0 = (pm[reference_idx] * u).sum(-1).mean(0)[None]
        pm = ((pm * u).sum(-1) - self.pm0)
        dp = torch.cat([dp3, dp8], 1)
        return dp, pm

    def train(self, dp, pm, y=None):
        self.dpmin = dp.min(0)[0][None]
        self.dpmax = dp.max(0)[0][None]
        self.pmmin = pm.min(0)[0][None]
        self.pmmax = pm.max(0)[0][None]

        dp, pm = self.normalize(dp, pm)
        self.trained_unsupervised = True

        if y is not None:
            self.dpclf = self.dpclf.fit(dp, y)
            self.pmclf = self.pmclf.fit(pm, y)
            self.trained_supervised = True

    def predict(self, dp, pm):
        if not self.trained_unsupervised:
            print("Not trained")
            return None
        _, pm = self.normalize(dp, pm)
        dpunsup, pmunsup = (dp ** 0.5).sum(1) / (dp > 0).sum(1), (1-pm).mean(1)
        dpsup, pmsup = None, None
        if self.trained_supervised:
            dp, _ = self.normalize(dp, pm)
            dpsup, pmsup = self.dpclf.predict_proba(dp)[:, 1], self.pmclf.predict_proba(pm)[:, 1]
        return dpunsup, pmunsup, dpsup, pmsup

class SSQAMut(SSQA):

    def __init__(self, model_ss, pattern=None, seq_hmm=None, ss_hmm3=None, ss_hmm8=None, name=""):
        super(SSQAMut, self).__init__(model_ss, pattern, seq_hmm, ss_hmm3, ss_hmm8, name)
        self.dpclf = RandomForestRegressor()
        self.pmclf = RandomForestRegressor()

    def predict(self, dp, pm):
        if not self.trained_unsupervised:
            print("Not trained")
            return None
        _, pm = self.normalize(dp, pm)
        dpunsup, pmunsup = (dp ** 0.5).sum(1) / (dp > 0).sum(1), pm.mean(1)
        dpsup, pmsup = None, None
        if self.trained_supervised:
            dp, _ = self.normalize(dp, pm)
            dpsup, pmsup = torch.tensor(self.dpclf.predict(dp)), torch.tensor(self.pmclf.predict(pm))
        return dpunsup, pmunsup, dpsup, pmsup

