import torch
import numpy as np

def overlap_to_one(x, p):
    x = x.expand(x.size(0), p.size(-1))
    inter_low = torch.max(x[0], p[0])
    inter_high = torch.min(x[1], p[1])
    union_low = torch.min(x[0], p[0])
    union_high = torch.max(x[1], p[1])

    return torch.clamp((inter_high - inter_low) / (union_high - union_low), 0, 1)

def to_bb(x):
    C = torch.tensor(range(x.size(-1))).view(1, 1, -1)
    z = x[:, :2].detach()
    z = z.round()
    return z

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