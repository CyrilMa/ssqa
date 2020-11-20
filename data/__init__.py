from .patterns_extraction import *
from .hmm_data import *
from .sequence_extraction import *
from .data_structure import *
import subprocess

def pfam_data(folder, filename):
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


def mutation_data(folder, filename):

    print("make HMM profile")
    subprocess.run(f'hhmake -i {folder}/{filename}.fasta -M 100', shell=True)

    print("build HMM profiles")
    build_profiles(folder)
