import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import pickle

import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .pattern import Matching
from data import SecondaryStructureRawDataset, collate_sequences
from utils import *
from ss_inference import NetSurfP2

inf = float("Inf")


class PatternMatchingInference(nn.Module):

    def __init__(self, model_ss3, Q, pi, seq_hmm, size=68, name=""):
        super(PatternMatchingInference, self).__init__()
        self.name = f"Pattern Inferer {name}"
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


def search_pattern(path, uniprot, seq_nat, c="A"):
    pdb_uniprot = pd.read_csv(f"{DATA}/cross/uniprot_pdb.csv", index_col=0)
    longest, patterns = 0, []
    for pdb in pdb_uniprot[pdb_uniprot.uni == uniprot].pdb.values:
        try:
            file_name = rcsb.fetch(pdb, "mmtf", biotite.temp_dir())
            mmtf_file = mmtf.MMTFFile()
            mmtf_file.read(file_name)
            # Transketolase homodimer
            ss_seq = np.array(list(mmtf_file["entityList"][0]["sequence"]))
            length, (m_nat, M_nat, m_mut, M_mut), _ = lcs_pattern(seq_nat, "".join(ss_seq))
            sse = mmtf_file["secStructList"]
            sse = np.array(sse[m_mut: M_mut + 1])
            length = len(sse)
            if length < longest:
                continue
            if length > longest:
                print(length)
                longest = length
                patterns = []
            sse = np.array([sec_struct_codes[code%8] for code in sse], dtype="U1")
            sse = np.array([dssp_to_abc[e] for e in sse], dtype="U1")
            sse = to_onehot([abc_codes[x] for x in sse], (None, 3))
            dss = (sse[1:] - sse[:-1])
            cls = to_onehot(np.where(dss == -1)[1], (None, 3)).T
            bbox = np.array([np.where(dss == 1)[0], np.where(dss == -1)[0], *cls]).T
            pat = np.argmax(bbox[:, 2:], 1)

            patterns.append((pat, m_nat, M_nat))
        except:
            continue
    ratio_covered = longest / len(seq_nat)
    if ratio_covered <= 0.9:
        pickle.dump((None, None, None, None), open(f"{path}/patterns.pkl", "wb"))
        return None, ratio_covered

    c_patterns, n_patterns, ms, Ms = [], [], [], []
    for pat,m_pat,M_pat in patterns:
        char_pat = "".join(["abc"[x] for x in pat])
        if len(char_pat):
            c_patterns.append(char_pat)
            n_patterns.append(list(pat))
            ms.append(m_pat)
            Ms.append(M_pat)
    max_occ, c_pattern, n_pattern, m_pat, M_pat = 0, None, None, None, None
    for c, n, m, M in zip(c_patterns, n_patterns, ms, Ms):
        n_occ = c_patterns.count(c)
        if n_occ > max_occ:
            max_occ = n_occ
            c_pattern, n_pattern = c, n
            m_pat, M_pat = m, M

    pickle.dump((n_pattern, c_pattern, m_pat, M_pat), open(f"{path}/patterns.pkl", "wb"))

    return (n_pattern, c_pattern, m_pat, M_pat), ratio_covered

def infer_pattern(path, indices = None):
    dataset = SecondaryStructureRawDataset(f"{path}/hmm.pkl")
    if indices is not None:
        dataset.primary = [x for i, x in enumerate(dataset.primary) if i in indices]
    loader = DataLoader(dataset, batch_size=16,
                        shuffle=False, drop_last=False, collate_fn=collate_sequences)

    Q = torch.load("Q.pt").float()
    pi = torch.load("pi.pt")[:, 0].float()
    seq_hmm = torch.tensor(dataset[0][0]).t()[20:]
    torch.save(seq_hmm, f"{path}/hmm.pt")
    _, size = seq_hmm.size()
    torch.cuda.empty_cache()

    model_ss3 = NetSurfP2(50, "nsp2")
    model_ss3 = model_ss3.to("cuda")
    model_ss3.load_state_dict(torch.load(f"{DATA}/secondary_structure/lstm_50feats.h5"))

    inferer = PatternMatchingInference(model_ss3, Q=Q, pi=pi,
                                       seq_hmm=seq_hmm, size=size)

    print("Inference Model build")
    c_patterns, n_patterns = dict(), dict()
    for batch_idx, data in enumerate(loader):
        x = torch.tensor(data[0])[None].permute(0, 2, 1).float()
        m = Matching(x)
        p = inferer(m, 1)
        for p_ in p:
            p_ = p_[:, 0]
            idx = torch.where(p_ < 3)[0]
            n_pattern = list(p_[idx].numpy())
            c_pattern = "".join("abc"[x] for x in p_[idx])
            if c_pattern not in c_patterns.keys():
                c_patterns[c_pattern] = 0
                n_patterns[n_pattern] = 0
            c_patterns[c_pattern] += 1
            n_patterns[n_pattern] += 1
    n_pattern = max(n_patterns, key= n_patterns.get)
    c_pattern = max(c_patterns, key= c_patterns.get)[0]
    print(f"Pattern Infered : {c_pattern}")
    pickle.dump((n_pattern, c_pattern, 0, x.size(-1)-1), open(f"{path}/patterns.pkl", "wb"))
    return n_pattern, c_pattern, 0, x.size(-1)-1


