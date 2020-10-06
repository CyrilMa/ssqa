import numpy as np

import torch
from torch.nn import functional as F

PROFILE_HEADER = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                  'M->M', 'M->I', 'M->D', 'I->M', 'I->I',
                  'D->M', 'D->D', 'Neff', 'Neff_I', 'Neff_D')  # yapf: disable
AMINO_ACIDS = AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = AA_IDS = {k: i for i, k in enumerate(AA)}
AA_MAT = None

# Dictionary to convert 'secStructList' codes to DSSP values
# https://github.com/rcsb/mmtf/blob/master/spec.md#secstructlist
sec_struct_codes = {0: "I",
                    1: "S",
                    2: "H",
                    3: "E",
                    4: "G",
                    5: "B",
                    6: "T",
                    7: "C"}

abc_codes = {"a": 0, "b": 1, "c": 2}
# Converter for the DSSP secondary pattern_matching elements
# to the classical ones
dssp_to_abc = {"I": "c",
               "S": "c",
               "H": "a",
               "E": "b",
               "G": "a",
               "B": "b",
               "T": "c",
               "C": "c"}


def to_onehot(a, shape):
    if shape[0] is None:
        shape = len(a), shape[1]
    onehot = np.zeros(shape)
    onehot[np.arange(len(a)), a] = 1
    return onehot


def to_aaindex(a):
    aaindex = np.zeros((a.shape[0], AA_MAT.shape[1]))
    for i, x in enumerate(a):
        if x == 26:
            aaindex[i] = np.zeros(AA_MAT.shape[1])
        else:
            aaindex[i] = AA_MAT[aa_dict[x]]
    return aaindex


def ss8_to_ss3(x):
    if x <= 2:
        return 0
    if x >= 5:
        return 2
    return 1


def to_bb(x):
    C = torch.tensor(range(x.size(-1))).view(1, 1, -1)
    z = x[:, :2].detach()
    z = z.round()
    return (z)


def overlap(ground_truth, predictions):
    union_low = torch.min(ground_truth[:, 0], predictions[:, 0])
    union_high = torch.max(ground_truth[:, 1], predictions[:, 1])
    inter_low = torch.max(ground_truth[:, 0], predictions[:, 0])
    inter_high = torch.min(ground_truth[:, 1], predictions[:, 1])
    return torch.clamp((inter_high - inter_low) / (union_high - union_low), 0, 1)


def L_box(ground_truth, predictions):
    # mask = (predictions[b_pos, 2:, p_pos].argmax(1) == ground_truth[b_pos, 2:, g_pos].argmax(1)).int().reshape(-1, 1)
    return F.smooth_l1_loss(predictions, ground_truth[:, :2], reduction="mean")


def L_conf(ground_truth, predictions, olap, b_pos, g_pos):
    return F.nll_loss(torch.log(predictions[b_pos, :, g_pos]), ground_truth[b_pos, :, g_pos].argmax(1), reduction="sum")


def to_seq(p, size):
    seq = np.zeros(size, dtype=int)
    for bbox in p.numpy().T[::-1]:
        seq[bbox[0]: bbox[1]] = bbox[2]
    return seq


def aa_acc(ground_truth, predictions, N):
    s, p = ground_truth, predictions
    p[2] = p[2:].argmax(0)
    p = p[:3].int().detach()
    s[2] = s[2:].argmax(0)
    s = s[:3].int().detach()
    seq_p = to_seq(p, N)
    seq_s = to_seq(s, N)
    return np.mean(seq_p == seq_s), seq_p, seq_s, p.t().numpy(), s.t().numpy()


def overlap_to_one(x, p):
    x = x.expand(x.size(0), p.size(-1))
    inter_low = torch.max(x[0], p[0])
    inter_high = torch.min(x[1], p[1])
    union_low = torch.min(x[0], p[0])
    union_high = torch.max(x[1], p[1])

    return torch.clamp((inter_high - inter_low) / (union_high - union_low), 0, 1)
