import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
from utils import *

def freq(freqstr):
    if freqstr == '*':
        return 0.
    p = 2 ** (int(freqstr) / -1000)
    assert 0 <= p <= 1.0
    return p


def process_msa(hhm_name):
    neff = None
    with open(hhm_name) as fp:
        for line in fp:
            if line[0:4] == 'NEFF' and neff is None:
                neff = float(line[4:].strip())
            if line[0:8] == 'HMM    A':
                header1 = line[7:].strip().split('\t')
                break
        header2 = next(fp).strip().split('\t')
        next(fp)

        profile = []
        for line in fp:
            if line[:2] == '//':
                break

            freqs = line.split(None, 2)[2].split('\t')[:20]
            features = {h: freq(i) for h, i in zip(header1, freqs)}
            assert len(freqs) == 20

            mid = next(fp)[7:].strip().split('\t')

            features.update({h: freq(i) for h, i in zip(header2, mid)})

            profile.append(features)
            next(fp)

        seqlen = len(profile)
        proflen = len(PROFILE_HEADER)
        profmat = np.zeros((seqlen, proflen), dtype='float')

        for i, pos in enumerate(profile):
            for j, key in enumerate(PROFILE_HEADER):
                profmat[i, j] = pos[key]

    return profmat

def build_profiles(folder, prefix = ""):
    hhm_name = f"{folder}/aligned.hhm"
    profile = process_msa(hhm_name)
    data = torch.load(f"{folder}/{prefix}data.pt")
    data["seq_hmm"] = torch.tensor(profile.T)
    torch.save(data, f"{folder}/{prefix}data.pt")