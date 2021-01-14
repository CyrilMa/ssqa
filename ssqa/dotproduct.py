import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F

inf = float("Inf")

from .pattern import Matching

class DotProduct(nn.Module):

    def __init__(self, model_ss, Q, seq_hmm, ss_hmm, size=68, name=""):
        r"""
        Class to perform dot product operations
        Args:
            model_ss (nn.Module): model to predict secondary structure
            Q (torch.Tensor): Emission matrix
            seq_hmm (torch.tensor): HMM profile of the sequence
            ss_hmm (torch.tensor): profile of the secondary structure
            size (int): max length of the sequence
            name (str): name of the model
        """
        super(DotProduct, self).__init__()
        self.name = f"Dot Product {name}"
        self.size = size
        self.ssn = Q.size(1)
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.Q = Q.view(1, *Q.size())
        self.model_ss = model_ss
        self.SEQ_HMM = seq_hmm
        self.SS_HMM = ss_hmm/(ss_hmm.pow(2).sum(1).pow(0.5)[:,None])

    def forward(self, m:Matching):
        batch_size = m.batch_size
        hmm, m.ls, idxs = self.hmm_(m)
        hmm = hmm.to(self.model_ss.device)
        ss = None
        if self.ssn == 3:
            m.ss = ss = F.softmax(self.model_ss(hmm)[2].cpu(),1).detach()
        if self.ssn == 8:
            m.ss = ss = F.softmax(self.model_ss(hmm)[1].cpu(),1).detach()
        ss_u = torch.zeros(batch_size, *self.SS_HMM.size()[1:])
        for i, (ss_,idx) in enumerate(zip(ss, idxs)):
            ss_ = ss_[:, :len(idx)]
            ss_u[i,:,idx] = ss_/(ss_.pow(2).sum(0).pow(0.5)[None])
        m.dp = (ss_u*self.SS_HMM).sum(1)
        return m

    def hmm_(self, m):
        batch_size, x = m.batch_size, m.x
        hmm = torch.zeros(batch_size, 20 + self.SEQ_HMM.size(0), self.size)
        ls = []
        idxs = []
        for i, x_ in zip(range(batch_size), x):
            idx = torch.where(x_[:20].sum(0) > 0)[0]
            n_idx = idx.size(0)
            ls.append(n_idx)
            hmm[i, :20, :n_idx] = x_[:20, idx]
            hmm[i, 20:, :n_idx] = self.SEQ_HMM[:, idx]
            idxs.append(idx)
        ls = torch.tensor(ls)
        return hmm, ls, idxs
