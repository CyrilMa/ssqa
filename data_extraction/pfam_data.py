import pandas as pd
import pickle
from tqdm import tqdm
import subprocess

from pattern_matching.utils import *

import biotite
import biotite.structure as struc

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf


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

