import numpy as np
import pandas as pd

from .utils import sec_struct_codes, dssp_to_abc, AMINO_ACIDS, PROFILE_HEADER

import biotite
import biotite.structure as struc

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# SEQUENCE EXTRACTION


def from_fasta_to_df(file):
    proteins_df = pd.DataFrame(columns=["aligned_seq", "seq"])
    with open(file, "r") as input_handle:
        for seq in SeqIO.parse(input_handle, "fasta"):
            sequence = str(seq.seq)
            if "X" in sequence:
                continue
            proteins_df.loc[seq.id] = [sequence, sequence.replace(".", "")]
    return proteins_df


def from_df_to_fasta(df, folder):
    records_aligned = []
    records_unaligned = []
    for ind, data in df.iterrows():
        records_aligned.append(SeqRecord(Seq(data.aligned_seq), id=ind))
        records_unaligned.append(SeqRecord(Seq(data.seq), id=ind))

    with open(f"{folder}/aligned.fasta", "w") as handle:
        SeqIO.write(records_aligned, handle, "fasta")
    with open(f"{folder}/unaligned.fasta", "w") as handle:
        SeqIO.write(records_unaligned, handle, "fasta")


# SECONDARY STRUCTURE EXRACTION

def fetch_PDB(pdb, c, start, end):
    # Fetch and load structure
    file_name = rcsb.fetch(pdb, "mmtf", biotite.temp_dir())
    mmtf_file = mmtf.MMTFFile()
    mmtf_file.read(file_name)
    array = mmtf.get_structure(mmtf_file, model=1)
    # Transketolase homodimer
    tk_dimer = array[struc.filter_amino_acids(array)]
    # Transketolase monomer
    tk_mono = tk_dimer[tk_dimer.chain_id == c]

    # The chain ID corresponding to each residue
    chain_id_per_res = array.chain_id[struc.get_residue_starts(tk_dimer)]
    sse = mmtf_file["secStructList"]
    sse = sse[sse != -1]
    sse = sse[:chain_id_per_res.shape[0]][chain_id_per_res == c]
    sse = np.array([sec_struct_codes[code] for code in sse if code != -1],
                   dtype="U1")
    sse = np.array([dssp_to_abc[e] for e in sse], dtype="U1")[start:end]
    return sse, tk_mono


# Parse HHM

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
