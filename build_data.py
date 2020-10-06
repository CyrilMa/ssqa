import pandas as pd
import pickle
import argparse

from torch.utils.data import DataLoader

from ss_inference.data import SecondaryStructureRawDataset, collate_sequences
from ss_inference.model import NetSurfP2

from pattern_matching.data_extraction import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="name of the dataset to compute data ")
parser.add_argument("steps", help="name of the dataset to compute data ")
args = parser.parse_args()
DATASET = args.dataset
STEPS = args.steps

if "1" in STEPS:
    extract_data(f"{DATA}/{DATASET}", "full.fasta")

if "2" in STEPS:
    structfam = get_structures(DATA, DATASET)
    pfam_seqs = pd.read_csv(f"{DATA}/{DATASET}/sequences.csv", index_col = 0, usecols = [0,2]).seq
    pfam_seqs = {k.split("/")[0]:v for k,v in pfam_seqs.items()}
    patterns, secondary_structure, c_patterns, n_patterns = build_patterns(pfam_seqs, structfam)

if "3" in STEPS:
    dataset = SecondaryStructureRawDataset(f"{DATA}/{DATASET}/hmm.pkl")
    loader = DataLoader(dataset, batch_size = 1,
                            shuffle=False, drop_last=False, collate_fn = collate_sequences)

    device = "cuda"

    model_ss3 = NetSurfP2(50)
    model_ss3 = model_ss3.to(device)
    model_ss3.load_state_dict(torch.load(f"{DATA}/secondary_structure/lstm_50feats.h5"))
    print(model_ss3)

    ss_prediction = model_ss3.predict(loader)
    pickle.dump(ss_prediction, open(f"{DATA}/{DATASET}/ss3.pkl", "wb"))
