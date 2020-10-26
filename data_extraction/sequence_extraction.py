import pandas as pd
import pickle

from random import shuffle
from pattern_matching.utils import *

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# SEQUENCE EXTRACTION

def from_fasta_to_df(folder, file, chunksize=5000):
    ids, seqs, aligned_seqs = [], [], []
    pd.DataFrame(columns=["aligned_seq", "seq", "length"]).to_csv(f"{folder}/sequences.csv")
    with open(file, "r") as input_handle:
        for i, seq in enumerate(SeqIO.parse(input_handle, "fasta")):
            seq = seq.upper()
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


def from_df_to_fasta(folder, file, chunksize=5000, prefix = ""):
    df_iter = pd.read_csv(file, index_col=0, chunksize=chunksize)
    for k, df in enumerate(df_iter):
        records_aligned = []
        records_unaligned = []
        for i, (ind, data) in enumerate(df.iterrows()):
            records_aligned.append(SeqRecord(Seq(data.aligned_seq), id=str(ind)))
            records_unaligned.append(SeqRecord(Seq(data.seq), id=str(ind)))
            print(f"Processing {i + chunksize * k} sequences ...", end="\r")
        with open(f"{folder}/{prefix}aligned.fasta", "a+") as handle:
            SeqIO.write(records_aligned, handle, "fasta")
        with open(f"{folder}/{prefix}unaligned.fasta", "a+") as handle:
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
