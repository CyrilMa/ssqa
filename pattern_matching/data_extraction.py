import pandas as pd
import pickle
from tqdm import tqdm
import subprocess

from random import shuffle
from .utils import *

import biotite
import biotite.structure as struc

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import netsurfp2 as nsp2


# SEQUENCE EXTRACTION

def from_fasta_to_df(folder, file, chunksize=5000):
    ids, seqs, aligned_seqs = [], [], []
    pd.DataFrame(columns=["aligned_seq", "seq", "length"]).to_csv(f"{folder}/sequences.csv")
    with open(file, "r") as input_handle:
        for i, seq in enumerate(SeqIO.parse(input_handle, "fasta")):
            if len(ids) >= chunksize:
                df = pd.DataFrame(index=ids)
                df["aligned_seq"] = aligned_seqs
                df["seq"] = seqs
                df["length"] = df.seq.apply(lambda seq: len(seq))
                ids, seqs, aligned_seqs = [], [], []
                df.to_csv(f"{folder}/sequences.csv", mode="a", header=False)
            aligned_seq = str(seq.seq)
            if "X" in aligned_seq:
                continue
            ids.append(seq.id)
            seq = "".join([c for c in aligned_seq if c in AA])
            seqs.append(seq), aligned_seqs.append(aligned_seq)
            print(f"Processing {i} sequences ...", end="\r")
    df = pd.DataFrame(index=ids)
    df["aligned_seq"] = aligned_seqs
    df["seq"] = seqs
    df["length"] = df.seq.apply(lambda seq: len(seq))
    df.to_csv(f"{folder}/sequences.csv", mode="a", header=False)


def from_df_to_fasta(folder, file, chunksize=5000):
    df_iter = pd.read_csv(file, index_col=0, chunksize=chunksize)
    for k, df in enumerate(df_iter):
        records_aligned = []
        records_unaligned = []
        for i, (ind, data) in enumerate(df.iterrows()):
            records_aligned.append(SeqRecord(Seq(data.aligned_seq), id=str(ind)))
            records_unaligned.append(SeqRecord(Seq(data.seq), id=str(ind)))
            print(f"Processing {i + chunksize * k} sequences ...", end="\r")
        with open(f"{folder}/aligned.fasta", "a+") as handle:
            SeqIO.write(records_aligned, handle, "fasta")
        with open(f"{folder}/unaligned.fasta", "a+") as handle:
            SeqIO.write(records_unaligned, handle, "fasta")


def build_protein_df(folder, filename, chunksize=5000):
    file = f"{folder}/{filename}"
    print("building sequences.csv")
    from_fasta_to_df(folder, file, chunksize=chunksize)
    print("building aligned.fasta, unaligned.fasta ...")
    from_df_to_fasta(folder, f"{folder}/sequences.csv", chunksize=chunksize)


# CLUSTERING/WEIGHTING/SPLITING

def cluster_weights(folder):
    clusters = pd.read_table(f"{folder}/tmp/clusters.tsv_cluster.tsv", names=["cluster", "id"]).set_index("id").cluster
    cluster_weights = 1 / clusters.value_counts()
    weights = [cluster_weights[c] for c in clusters]
    pickle.dump(list(clusters.index), open(f"{folder}/index.pkl", "wb"))
    pickle.dump(list(weights), open(f"{folder}/weights.pkl", "wb"))
    return pd.Series(data=weights, index=clusters.index)


def split_train_val_set(folder, ratio=0.1):
    clusters = pd.read_table(f"{folder}/tmp/clusters.tsv_cluster.tsv", names=["clusters", "id"]).set_index(
        "id").clusters
    max_size = ratio * len(clusters)
    val = []
    unique_clusters = list(clusters.unique())
    shuffle(unique_clusters)
    for c in unique_clusters:
        val += list(clusters[clusters == c].index)
        if len(val) > max_size:
            break
    is_val = [int(c in val) for c in clusters.index]
    pickle.dump(is_val, open(f"{folder}/is_val.pkl", "wb"))
    return pd.Series(data=is_val, index=clusters.index)


# HMM PROFILES

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


def build_profiles(folder, chunksize=5000):
    with open(f"{folder}/aligned.fasta") as f:
        protlist = nsp2.parse_fasta(f)
    hhm_name = f"{folder}/aligned.hhm"
    dataset = dict()
    for k, (_, v) in tqdm(protlist.items()):
        prof = process_msa(v, hhm_name)["profile"]
        idx = np.where(prof[:, :20].sum(1))
        dataset[k] = prof[idx]
    pickle.dump(dataset, open(f"{folder}/hmm.pkl", "ab"))


# FULL EXTRACTION

def extract_data(folder, filename):
    build_protein_df(f"{folder}", filename)

    print("building clusters with MMSEQS")
    subprocess.run(
        f'mmseqs easy-cluster "{folder}/unaligned.fasta" "{folder}/tmp/clusters.tsv" "{folder}/tmp" --min-seq-id 0.7',
        shell=True)

    print("computing cluster weights")
    cluster_weights(folder)

    print("split between training and validation set")
    split_train_val_set(folder)

    print("make HMM profile")
    subprocess.run(f'hhmake -i {folder}/aligned.fasta -M 100', shell=True)

    print("build HMM profiles")
    build_profiles(folder)


# SECONDARY STRUCTURE EXTRACTION

def get_structures(data, dataset):
    pdb_uniprot = pd.read_csv(f"{data}/cross/uniprot_pdb.csv", index_col=0)
    mapping = pd.read_table(f"{data}/cross/pdb_to_pfam.txt")
    all_names = [x.split("/")[0] for x in pickle.load(open(f"{data}/{dataset}/index.pkl", "rb"))]

    mapping["PFAM_ID"] = mapping.PFAM_ACC.apply(lambda x: x.split(".")[0])
    fam = mapping[mapping.PFAM_ID == dataset.split("/")[0]]

    structfam = set()
    counts = dict()
    for x in fam.itertuples():
        keys = list(pdb_uniprot[pdb_uniprot.pdb == x.PDB_ID].uni)
        for key in keys:
            if key in all_names:
                structfam.add((key, x.PDB_ID, x.CHAIN_ID))
                if key in counts.keys():
                    counts[key] += 1
                else:
                    counts[key] = 1
    structfam2 = set()
    for x in structfam:
        structfam2.add((*x, 1 / counts[x[0]]))
    return structfam2


INDEL, MISS = 3, 50


def lcs(X, Y):
    m, n = len(X), len(Y)
    L = np.zeros((m + 1, n + 1))

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i, j] = max(1, L[i - 1, j - 1] + 1)
            else:
                L[i, j] = max(0, L[i - 1, j] - INDEL, L[i, j - 1] - INDEL, L[i - 1, j - 1] - MISS)

    # Following code is used to print LCS

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i, j = np.unravel_index(L.argmax(), L.shape)
    posX, posY = [], []
    while i > 0 and j > 0:
        if L[i, j] == 0:
            break
        if X[i - 1] == Y[j - 1]:
            i -= 1;
            j -= 1
            posX.append(i);
            posY.append(j)
            continue
        insert, delete, miss = L[i - 1, j] - INDEL, L[i, j - 1] - INDEL, L[i - 1, j - 1] - MISS
        if insert > delete and insert > miss:
            i -= 1
        elif delete > miss:
            j -= 1
        else:
            i -= 1;
            j -= 1
    posX.sort(), posY.sort()
    return len(posX), posX, posY


def pfam_matching(pfam_seqs, S, uniprot_id):
    best_matching = None
    longest = 0
    pos = None
    for key, val in pfam_seqs.items():
        if key.split("/")[0] == uniprot_id:
            length, posX, posY = lcs(S, val)
            if length > longest:
                best_matching = key, length, posX, posY
                longest = length
    return best_matching


def fetch_PDB(pfam_seqs, pdb, c, uniprot_id):
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
    chain_idx = np.where(chain_id_per_res == c)[0]
    seq = np.array(list(mmtf_file["entityList"][0]["sequence"]))[chain_idx]

    key, length, pos_X, pos_Y = pfam_matching(pfam_seqs, seq, uniprot_id)
    sse = mmtf_file["secStructList"]
    sse = sse[:chain_id_per_res.shape[0]][chain_id_per_res == c]
    idx = np.where(sse == -1)[0]
    while len(idx) > 0:
        sse[idx] = sse[idx - 1]
        idx = np.where(sse == -1)[0]
    sse = np.array([sec_struct_codes[code] for code in sse], dtype="U1")
    sse = np.array([dssp_to_abc[e] for e in sse], dtype="U1")[pos_X]
    return key, pos_X, pos_Y, sse


def build_patterns(pfam_seqs, structfam):
    patterns = dict()
    secondary_structure = dict()
    for uniprot_id, pdb, c, weight in tqdm(structfam):
        try:
            key, pos_X, pos_Y, sse = fetch_PDB(pfam_seqs, pdb, c, uniprot_id)
            ss = to_onehot([abc_codes[x] for x in sse], (None, 3))
            ss = np.pad(ss, ((1, 1), (0, 0)), "constant")
            dss = (ss[1:] - ss[:-1])
            cls = to_onehot(np.where(dss == -1)[1], (None, 3)).T
            bbox = np.array([np.where(dss == 1)[0], np.where(dss == -1)[0], *cls]).T
            struct = np.array([abc_codes[x] for x in sse])
            secondary_number = np.zeros(len(struct), dtype=int)
            for i, x in enumerate(bbox):
                secondary_number[int(x[0]):int(x[1])] = i
            patterns[pdb] = np.argmax(bbox[:, 2:], 1)
            secondary_structure[pdb] = key, struct, secondary_number, np.array(pos_Y), weight
        except:
            ()
    c_patterns, n_patterns = [], []
    for k, pat in patterns.items():
        char_pat = "".join(["abc"[x] for x in pat])
        if len(char_pat) and char_pat not in c_patterns:
            c_patterns.append(char_pat)
            n_patterns.append(list(pat))

    return patterns, secondary_structure, c_patterns, n_patterns