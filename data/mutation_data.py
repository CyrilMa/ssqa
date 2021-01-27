import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import subprocess

from utils import *
from ssqa import search_pattern, infer_pattern
from .hmm_data import build_profiles
from .sequence_extraction import from_df_to_fasta, from_df_to_data, from_fasta_to_df

# Family


def replace(seq, mutations):
    mutations = mutations.split(",")
    for mut in mutations:
        if len(mut) == 0:
            continue
        i = int(mut[1:-1])
        end = mut[-1]
        seq = seq[:i]+end+seq[i+1:]
    return seq


def run_family(path, family, uniprot):
    from_fasta_to_df(f"{path}/{family}", f"{path}/{family}/{family}.fasta", chunksize=5000)
    from_df_to_fasta(f"{path}/{family}", f"{path}/{family}/sequences.csv")
    from_df_to_data(f"{path}/{family}", f"{path}/{family}/sequences.csv")

    print("make HMM profile")
    subprocess.run(f'hhmake -i {path}/{family}/aligned.fasta -M 100', shell=True)

    nat_df = pd.read_csv(f"{path}/{family}/sequences.csv")
    seq_nat = nat_df.loc[0].seq
    build_profiles(f"{path}/{family}")
    pattern, ratio_covered = search_pattern(f"{path}/{family}", uniprot, seq_nat)
    if pattern is None:
        pattern, ratio_covered = infer_pattern(f"{path}/{family}")
    return pattern, ratio_covered

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
    from_df_to_data(f"{path}/{family}", f"{path}/{family}/{name_dataset}_mutation_sequences.csv",
                     prefix=f"{name_dataset}_")
    build_profiles(f"{path}/{family}", prefix=f"{name_dataset}_")

