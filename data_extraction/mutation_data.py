import pandas as pd
import pickle
import subprocess

import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

from .hmm_data import build_profiles
from .sequence_extraction import from_df_to_fasta, from_fasta_to_df
from .utils import *

# Family

def lcs(X, Y):
    m, n = len(X), len(Y)
    L = np.zeros((m + 1, n + 1))

    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i - 1] == Y[j - 1] or (Y[j-1] == "-" and L[i - 1, j - 1]>0):
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = 0

    i, j = np.unravel_index(L.argmax(), L.shape)
    posX, posY = [], []
    while i > 0 and j > 0:
        if L[i, j] == 0:
            break
        if X[i - 1] == Y[j - 1] or Y[j-1] == "-":
            i -= 1
            j -= 1
            posX.append(i)
            posY.append(j)
            continue
    posX.sort(), posY.sort()
    return len(posX), (min(posX), max(posX), min(posY), max(posY)), L

INDEL, MISS = 3, 10

def lcs_pattern(X, Y):
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
    return len(posX), (min(posX), max(posX), min(posY), max(posY)), L


def replace(seq, mutations):
    mutations = mutations.split(",")
    for mut in mutations:
        if len(mut) == 0:
            continue
        i = int(mut[1:-1])
        end = mut[-1]
        seq = seq[:i]+end+seq[i+1:]
    return seq

def find_patterns(uniprot, seq_nat, c="A"):
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
    if longest < 0.7 * len(seq_nat):
        return None, longest/len(seq_nat)
    return patterns, longest/len(seq_nat)

def clean_family(path, family):
    #TODO
    return

def run_family(path, family, uniprot):
    from_fasta_to_df(f"{path}/{family}", f"{path}/{family}/{family}.fasta", chunksize=5000)
    from_df_to_fasta(f"{path}/{family}", f"{path}/{family}/sequences.csv")

    print("make HMM profile")
    subprocess.run(f'hhmake -i {path}/{family}/aligned.fasta -M 100', shell=True)

    nat_df = pd.read_csv(f"{path}/{family}/sequences.csv")
    seq_nat = nat_df.loc[0].seq
    del nat_df

    patterns, ratio_covered = find_patterns(uniprot, seq_nat)
    if patterns is None:
        pickle.dump((None, None, None, None), open(f"{path}/{family}/patterns.pkl", "wb"))
        subprocess.run(f'rm -rf {path}/{family}/unaligned.fasta', shell=True)
        return False, ratio_covered
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

    pickle.dump((n_pattern, c_pattern, m_pat, M_pat), open(f"{path}/{family}/patterns.pkl", "wb"))
    subprocess.run(f'rm -rf {path}/{family}/unaligned.fasta', shell=True)
    subprocess.run(f'rm -rf {path}/{family}/aligned.fasta', shell=True)
    return True, ratio_covered

# Dataset

def seq_to_align(seq, no_gaps, N):
    aseq = np.array(["-"]*N)
    aseq[no_gaps] = [x for x in seq]
    return "".join(aseq)

def run_dataset(path, family, name_dataset):
    nat_df = pd.read_csv(f"{path}/{family}/sequences.csv")
    aligned_seq_nat = nat_df.loc[0].aligned_seq
    N = len(aligned_seq_nat)
    no_gaps = np.array([i for i,x in enumerate(aligned_seq_nat) if x != "-"])
    seq_nat = nat_df.loc[0].seq
    del nat_df

    mut_df = pd.read_csv(f"{path}/{family}/{family}_{name_dataset}.csv", comment="#", sep=";")
    positions = dict()
    for mutation in mut_df.mutant:
        muts = mutation.split(",")
        for mut in muts:
            if len(mut) <= 2:
                continue
            else:
                positions[int(mut[1:-1])] = mut[0]
    seq_mut = "".join([positions.get(i, "-") for i in range(max(positions.keys()) + 1)])
    _, (m_nat, M_nat, m_mut, M_mut), _ = lcs(seq_nat, seq_mut)

    mut_df["name"] = mut_df.mutant.apply(lambda x: ",".join([(x_[0] + str(int(x_[1:-1]) - m_mut + m_nat) + x_[-1]) for x_ in x.split(",") if ((int(x_[1:-1]) <= M_mut) and (int(x_[1:-1]) >= m_mut))])  if x.lower() != "wt" else "WT")
    mut_df["name"] = mut_df["name"].fillna("WT")
    mut_df["seq"] = seq_nat
    mut_df["seq"] = mut_df.apply(lambda x: replace(x.seq, x["name"]) if x.mutant.upper() != "WT" else x.seq, axis=1)
    mut_df["aligned_seq"] = mut_df["seq"].apply(lambda x : seq_to_align(x, no_gaps, N))
    mut_df = mut_df.set_index("name")

    mut_df.to_csv(f"{path}/{family}/{name_dataset}_mutation_sequences.csv")
    from_df_to_fasta(f"{path}/{family}", f"{path}/{family}/{name_dataset}_mutation_sequences.csv",
                     prefix=f"{name_dataset}_")
    build_profiles(f"{path}/{family}", prefix=f"{name_dataset}_")


def clean_dataset(path, family, dataset_name):
    # TODO
    return
