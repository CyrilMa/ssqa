import pandas as pd
import argparse

from pattern_matching.data_extraction import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="name of the dataset to compute data ")
args = parser.parse_args()
DATASET = args.dataset

extract_data(f"{DATA}/{DATASET}", "full.fasta")
structfam = get_structures(DATA, DATASET)
pfam_seqs = pd.read_csv(f"{DATA}/{DATASET}/sequences.csv", index_col = 0, usecols = [0,2]).seq
pfam_seqs = {k.split("/")[0]:v for k,v in pfam_seqs.items()}
patterns, secondary_structure, c_patterns, n_patterns = build_patterns(pfam_seqs, structfam)