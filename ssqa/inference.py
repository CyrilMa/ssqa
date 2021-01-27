import sys, os ,pickle
ROOT = os.path.dirname(os.getcwd())
sys.path.append(ROOT)
from config import *

import pandas as pd

import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .pattern import Matching
from data import *
from utils import *
from ss_inference import NetSurfP2

inf = float("Inf")


class PatternMatchingInference(nn.Module):
    r"""
    Class to infer pattern if not available
    Args:
        model_ss (nn.Module): model to predict secondary structure
        Q (torch.Tensor): Emission matrix
        pi (torch.tensor): initial state
        seq_hmm (torch.tensor): HMM profile of the sequence
        size (int): max length of the sequence
        name (str): name of the model
    """

    def __init__(self, model_ss, Q, pi, seq_hmm, size=68, name=""):
        super(PatternMatchingInference, self).__init__()
        self.name = f"Pattern Inferer {name}"
        self.size = size
        Q = torch.tensor(Q[:, :, :size + 1, :size + 1]).float()
        self.ssn = Q.size(1) - 1
        self.Q = Q.view(1, *Q.size())
        self.pi = pi.view(1, *pi.size())
        self.model_ss = model_ss
        self.SEQ_HMM = seq_hmm
        self.do_match = True

    def __repr__(self):
        return self.name

    def forward(self, m: Matching, T):
        hmm, m.ls = self.hmm_(m)
        hmm = hmm.to(self.model_ss.device)
        if self.ssn == 3:
            m.ss = F.softmax(self.model_ss(hmm)[2].cpu(), 1).detach() + 1e-3
        if self.ssn == 8:
            m.ss = F.softmax(self.model_ss(hmm)[1].cpu(), 1).detach() + 1e-3
        m.P = self.P_(m).float()
        m.a = self.max_alpha(m, T)
        return self.argmax_alpha(m, T)

    def max_alpha(self, m, T):
        batch_size, P, ls = m.batch_size, m.P, m.ls
        alpha, norm = [], []
        a_ = -torch.ones(batch_size, self.ssn + 1, self.size + 1) * inf
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
        c_star, l_star, t_star = torch.ones(batch_size).long() * 3, torch.zeros(batch_size), ls
        max_pattern = [torch.cat([c_star.view(-1, 1), l_star.view(-1, 1), t_star.view(-1, 1)], 1)]

        for a in alpha.transpose(1, 0)[1:]:
            a_ = (P[torch.arange(batch_size), :, c_star, :, t_star] + T * self.Q.expand(batch_size, -1, -1, -1, -1)[
                                                                          torch.arange(batch_size), :, c_star, :,
                                                                          t_star] + a).view(batch_size, -1).argmax(-1)
            c_star, l_star, t_star = a_ // size, a_ % size - t_star, a_ % size
            max_pattern.append(torch.cat([c_star.view(-1, 1), l_star.view(-1, 1), t_star.view(-1, 1)], 1))
        max_pattern = torch.cat([p[:, None] for p in max_pattern[::-1]], 1)
        max_pattern[:,:-1,1] = max_pattern[:,1:,1]
        max_pattern[:,-1,1] = 0
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
        batch_size, C, ls = m.batch_size, torch.log(m.ss), m.ls
        P = torch.zeros(batch_size, self.ssn+1, self.size + 1, self.size + 1)
        for i in range(self.size + 1):
            P[:, :self.ssn, i, :i + 1] = -inf
            if i == self.size:
                break
            P[:, :self.ssn, :i + 1, i + 1:] += C[:, :, i].view(batch_size, self.ssn, 1, 1)
        P[:, -1] = -inf
        P[:, -1, ls, ls] = 0
        return P[:, None, :, :, :]


def search_pattern(path, uniprot, seq_nat):
    r"""
    Search a pattern with PDB
    Args:
        path (str): path to save data
        uniprot (str): uniprot id of the search sequence
        seq_nat (str): raw sequences for a better alignment of the pattern with the sequence
    """
    pdb_uniprot = pd.read_csv(f"{CROSS}/uniprot_pdb.csv", index_col=0)
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
            if max(sse) == -1:
                continue
            if length < longest:
                continue
            if length > longest:
                longest = length
                patterns = []
            sse = np.array([pdb_codes[code%8] for code in sse], dtype="U1")

            sse8 = to_onehot([dssp_codes[x] for x in sse], (None, 8))
            dss8 = (sse8[1:] - sse8[:-1])
            cls = to_onehot(np.where(dss8 == -1)[1], (None, 8)).T
            bbox = np.array([np.where(dss8 == 1)[0], np.where(dss8 == -1)[0], *cls]).T
            pat8 = np.argmax(bbox[:, 2:], 1)

            sse3 = to_onehot([abc_codes[dssp_to_abc[x]] for x in sse], (None, 3))
            dss3 = (sse3[1:] - sse3[:-1])
            cls = to_onehot(np.where(dss3 == -1)[1], (None, 3)).T
            bbox = np.array([np.where(dss3 == 1)[0], np.where(dss3 == -1)[0], *cls]).T
            pat3 = np.argmax(bbox[:, 2:], 1)
            patterns.append((list(pat3), list(pat8)))
        except:
            continue
    ratio_covered = longest / len(seq_nat)
    if ratio_covered <= 0.9:
        push(f"{path}/data.pt", "pattern", (None,None,None,None))
        return None, ratio_covered
    c_patterns3, n_patterns3, c_patterns8, n_patterns8 = [], [], [], []
    for pat3, pat8 in patterns:
        if len(pat3) == 0:
            continue
        if pat3[0] != 2:
            pat3 = [2] + pat3
        if pat3[-1] != 2:
            pat3 = pat3 + [2]
        if pat8[0] != 7:
            pat8 = [7] + pat8
        if pat8[-1] != 7:
            pat8 = pat8 + [7]
        char_pat8 = "".join([sec_struct_codes[x] for x in pat8])
        char_pat3 = "".join(["abc"[x] for x in pat3])
        c_patterns8.append(char_pat8)
        n_patterns8.append(list(pat8))
        c_patterns3.append(char_pat3)
        n_patterns3.append(list(pat3))
    max_occ, c_pattern3, n_pattern3, c_pattern8, n_pattern8 = 0, None, None, None, None
    for c3, n3, c8, n8 in zip(c_patterns3, n_patterns3, c_patterns8, n_patterns8):
        n_occ = c_patterns8.count(c8)
        if n_occ > max_occ:
            max_occ = n_occ
            c_pattern3, n_pattern3 = c3, n3
            c_pattern8, n_pattern8 = c8, n8
    push(f"{path}/data.pt", "pattern", (c_pattern3, n_pattern3, c_pattern8, n_pattern8))

    return (c_pattern3, n_pattern3, c_pattern8, n_pattern8), ratio_covered


def infer_pattern(path, indices=None):
    r"""
    infer a pattern if not available
    Args:
        path (str): path to load and save data
        indices (list): reference sequences in fasta file
    """
    if indices is None:
        indices = list(range(128))
    dataset = SSQAData_QA(f"{path}/data.pt")

    Q3 = torch.load(f"{UTILS}/Q3.pt").float()
    pi3 = torch.load(f"{UTILS}/pi3.pt")[:, 0].float()
    Q8 = torch.load(f"{UTILS}/Q8.pt").float()
    pi8 = torch.load(f"{UTILS}/pi8.pt")[:, 0].float()
    seq_hmm = dataset.seq_hmm
    _, size = seq_hmm.size()
    torch.cuda.empty_cache()

    model_ss = NetSurfP2(50, "nsp2")
    model_ss = model_ss.to("cuda")
    model_ss.load_state_dict(torch.load(f"{UTILS}/nsp_50feats.h5"))

    inferer3 = PatternMatchingInference(model_ss, Q=Q3, pi=pi3,
                                        seq_hmm=seq_hmm, size=size)
    inferer8 = PatternMatchingInference(model_ss, Q=Q8, pi=pi8,
                                        seq_hmm=seq_hmm, size=size)

    print("Inference Model build")
    c_patterns3, n_patterns3 = dict(), dict()
    c_patterns8, n_patterns8 = dict(), dict()
    X = torch.cat([data[None] for data in dataset], 0)
    X = X[indices]
    batch_size = 16
    N = len(X)
    for batch_idx in range(N // batch_size + 1):
        x = X[batch_idx * batch_size: (batch_idx + 1) * batch_size].float()
        m = Matching(x)
        p3 = inferer3(m, 3)
        p8 = inferer8(m, 3)
        for p3_, p8_ in zip(p3, p8):
            p_ = p8_[:, 0].int()
            idx = torch.where(p_ < 8)[0]
            c_pattern = "".join(sec_struct_codes[x] for x in p_[idx].numpy())
            if c_pattern not in c_patterns8.keys():
                c_patterns8[c_pattern] = 0
            c_patterns8[c_pattern] += 1

            p_ = p3_[:, 0].int()
            idx = torch.where(p_ < 3)[0]
            c_pattern = "".join("abc"[x] for x in p_[idx].numpy())
            if c_pattern not in c_patterns3.keys():
                c_patterns3[c_pattern] = 0
            c_patterns3[c_pattern] += 1

    print(c_patterns8)
    print(c_patterns3)

    c_pattern3 = max(c_patterns3, key=c_patterns3.get)
    c_pattern8 = max(c_patterns8, key=c_patterns8.get)
    n_pattern3 = [abc_codes[x] for x in c_pattern3]
    n_pattern8 = [dssp_codes[x] for x in c_pattern8]

    print(f"Pattern Infered : ss3 = {c_pattern3}, ss8 = {c_pattern8} ")
    push(f"{path}/data.pt", "pattern", (c_pattern3, n_pattern3, c_pattern8, n_pattern8))
    return (n_pattern3, c_pattern3, n_pattern8, c_pattern8), 1.


def build_piQT3():
    data = pickle.load(open(f"{UTILS}/training_set", "rb"))

    bboxes = []
    for x, sse in data.values():
        ss = to_onehot(np.array([ss8_to_ss3(x_) for x_ in x[:, 57:65].argmax(-1)]), (None, 3))
        ss = np.pad(ss, ((1, 1), (0, 0)), "constant")
        dss = (ss[1:] - ss[:-1])
        cls = to_onehot(np.where(dss == -1)[1], (None, 3)).T
        bbox = np.array([np.where(dss == 1)[0], np.where(dss == -1)[0], *cls]).T
        bboxes.append(bbox)

    pi = np.zeros(4)
    pi[:-1] = np.array([x[0, 2:] for x in bboxes] + [x[-1, 2:] for x in bboxes]).sum(0)

    T_ = [np.append(x[:-1, 2:], x[1:, 2:], axis=1) for x in bboxes]
    T_ = np.concatenate(T_, axis=0)

    T = np.zeros((4, 4))
    for i in range(3):
        t = T_[np.where(T_[:, i] == 1)[0], 3:]
        T[i, :3] = t.sum(0)
        T[i, 3] = pi[i]
        T[i] /= T[i].sum()
    T[3, 3] = 1
    T = T.reshape(4, 4, 1, 1)

    bboxes_sp = [np.concatenate([x[np.where(x[:, i] == 1)[0]][:, :2] for x in bboxes], axis=0) for i in
                 range(2, len(bboxes[0][0]))]

    Q_ = np.zeros((3, 100))
    for i in range(3):
        Q_[i] = plt.hist(bboxes_sp[i][:, 1] - bboxes_sp[i][:, 0], bins=np.linspace(0, 100, 101) - 0.5)[0]
        Q_[i] /= Q_[i].sum()

    Q = np.zeros((4, 500, 500))

    for j in range(500):
        Q[:-1, j, j:min(j + 100, 500)] = Q_[:, :min(j + 100, 500) - j]
    Q[-1, np.arange(500), np.arange(500)] = 1

    Q = Q.reshape((1, 4, 500, 500))

    Q = torch.tensor(np.log(Q) + np.log(T))
    pi = torch.tensor(np.log(pi / pi.sum()).reshape(-1, 1))
    torch.save(pi, f"{UTILS}/pi3.pt")
    torch.save(Q, f"{UTILS}/Q3.pt")

def build_piQT8():
    data = pickle.load(open(f"{UTILS}//training_set", "rb"))

    bboxes = []
    for x, sse in data.values():
        ss = x[:, 57:65]
        ss = np.pad(ss, ((1, 1), (0, 0)), "constant")
        dss = (ss[1:] - ss[:-1])
        cls = to_onehot(np.where(dss == -1)[1], (None, 8)).T
        bbox = np.array([np.where(dss == 1)[0], np.where(dss == -1)[0], *cls]).T
        bboxes.append(bbox)

    pi = np.zeros(9)
    pi[:-1] = np.array([x[0, 2:] for x in bboxes] + [x[-1, 2:] for x in bboxes]).sum(0)

    T_ = [np.append(x[:-1, 2:], x[1:, 2:], axis=1) for x in bboxes]
    T_ = np.concatenate(T_, axis=0)

    T = np.zeros((9, 9))
    for i in range(8):
        t = T_[np.where(T_[:, i] == 1)[0], 8:]
        T[i, :8] = t.sum(0)
        T[i, 8] = pi[i]
        T[i] /= T[i].sum()
    T[8, 8] = 1
    T = T.reshape(9, 9, 1, 1)

    bboxes_sp = [np.concatenate([x[np.where(x[:, i] == 1)[0]][:, :2] for x in bboxes], axis=0) for i in
                 range(2, len(bboxes[0][0]))]

    Q_ = np.zeros((8, 100))
    for i in range(8):
        Q_[i] = plt.hist(bboxes_sp[i][:, 1] - bboxes_sp[i][:, 0], bins=np.linspace(0, 100, 101) - 0.5)[0]
        Q_[i] /= Q_[i].sum()

    Q = np.zeros((9, 500, 500))

    for j in range(500):
        Q[:-1, j, j:min(j + 100, 500)] = Q_[:, :min(j + 100, 500) - j]
    Q[-1, np.arange(500), np.arange(500)] = 1

    Q = Q.reshape((1, 9, 500, 500))

    Q = torch.tensor(np.log(Q) + np.log(T))
    pi = torch.tensor(np.log(pi / pi.sum()).reshape(-1, 1))
    torch.save(pi, f"{UTILS}/pi8.pt")
    torch.save(Q, f"{UTILS}/Q8.pt")