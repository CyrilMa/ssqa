import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
from tqdm import tqdm
import netsurfp2 as nsp2

from utils import *



def process_msa(seq, hhm_name):
    with open(hhm_name) as fp:
        res = parse_hhm(fp, seq=seq)
    return res


def freq(freqstr):
    if freqstr == '*':
        return 0.
    p = 2 ** (int(freqstr) / -1000)
    assert 0 <= p <= 1.0
    return p


def parse_hhm(hhmfp, seq=None):
    neff = None
    for line in hhmfp:
        if line[0:4] == 'NEFF' and neff is None:
            neff = float(line[4:].strip())
        if line[0:8] == 'HMM    A':
            header1 = line[7:].strip().split('\t')
            break

    header2 = next(hhmfp).strip().split('\t')
    next(hhmfp)

    hh_seq = []
    profile = []
    for line in hhmfp:
        if line[:2] == '//':
            break
        aa = line[0]
        hh_seq.append(aa)

        freqs = line.split(None, 2)[2].split('\t')[:20]
        features = {h: freq(i) for h, i in zip(header1, freqs)}
        assert len(freqs) == 20

        mid = next(hhmfp)[7:].strip().split('\t')

        features.update({h: freq(i) for h, i in zip(header2, mid)})

        profile.append(features)
        next(hhmfp)

    hh_seq = ''.join(hh_seq)
    seq = seq or hh_seq
    profile = vectorize_profile(profile, seq, hh_seq)

    return {'profile': profile,
            'neff': neff,
            'header': header1 + header2, }


def vectorize_profile(profile,
                      seq,
                      hh_seq,
                      amino_acids=None,
                      profile_header=None):
    if profile_header is None:
        profile_header = PROFILE_HEADER

    if amino_acids is None:
        amino_acids = AMINO_ACIDS

    seqlen = len(seq)
    aalen = len(amino_acids)
    proflen = len(profile_header)
    profmat = np.zeros((seqlen, aalen + proflen + 1), dtype='float')

    for i, aa in enumerate(seq):
        aa_idx = amino_acids.find(aa)
        if aa_idx > -1:
            profmat[i, aa_idx] = 1.

    if len(profile) == len(seq):
        for i, pos in enumerate(profile):
            for j, key in enumerate(profile_header, aalen):
                profmat[i, j] = pos[key]
    else:
        hh_index = -1
        for i, restype in enumerate(seq):
            if restype != 'X':
                hh_index += 1
                assert restype == hh_seq[hh_index]

            if hh_index >= 0:
                for j, key in enumerate(profile_header, aalen):
                    profmat[i, j] = profile[hh_index][key]

    profmat[:, -1] = 1.
    return profmat


def build_profiles(folder, prefix = "", idxs = None):
    with open(f"{folder}/{prefix}aligned.fasta") as f:
        protlist = nsp2.parse_fasta(f)
    if idxs is not None:
        protlist = {k:v for i,(k,v) in enumerate(protlist.items()) if i in idxs}
    hhm_name = f"{folder}/aligned.hhm"
    dataset = dict()
    for k, (_, v) in tqdm(protlist.items()):
        prof = process_msa(v, hhm_name)["profile"]
        idx = np.where(prof[:, :20].sum(1))
        dataset[k] = prof[idx]
    pickle.dump(dataset, open(f"{folder}/{prefix}hmm.pkl", "ab"))
