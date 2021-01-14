import pandas as pd
import pickle
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader

from ss_inference import NetSurfP2
from data import *
from config import *
from utils import *

DATA = PFAM_DATA

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="name of the dataset to compute data ")
parser.add_argument("steps", help="name of the dataset to compute data ")
args = parser.parse_args()
DATASET = args.dataset
STEPS = args.steps

if "1" in STEPS:
    pfam_data(f"{DATA}/{DATASET}", "full.fasta")

if "2" in STEPS:
    structfam = get_structures(DATA, DATASET)
    pfam_seqs = pd.read_csv(f"{DATA}/{DATASET}/sequences.csv", index_col = 0, usecols = [0,2]).seq
    pfam_seqs = {k.split("/")[0]:v for k,v in pfam_seqs.items()}
    patterns, secondary_structure, c_patterns, n_patterns = build_patterns(pfam_seqs, structfam)

dataset = SecondaryStructureRawDataset(f"{DATA}/{DATASET}/hmm.pkl")

if "3" in STEPS:
    print("Secondary structure")
    loader = DataLoader(dataset, batch_size = 1,
                            shuffle=False, drop_last=False, collate_fn = collate_sequences)

    device = "cuda"

    model_ss3 = NetSurfP2(50)
    model_ss3 = model_ss3.to(device)
    model_ss3.load_state_dict(torch.load(f"{DATA}/secondary_structure/lstm_50feats.h5"))
    print(model_ss3)

    others, ss8, ss3 = model_ss3.predict(loader)
    pickle.dump(ss3, open(f"{DATA}/{DATASET}/ss3.pkl", "wb"))
    pickle.dump(ss8, open(f"{DATA}/{DATASET}/ss8.pkl", "wb"))
    pickle.dump(others, open(f"{DATA}/{DATASET}/others.pkl", "wb"))

if "4" in STEPS:

    raw_sequences = np.array(list(pd.read_csv(f"{DATA}/{DATASET}/sequences.csv").aligned_seq.apply(
        lambda s_: np.array([AA_IDS[c] + 1 if c in AA_IDS.keys() else 0 for c in s_]))))

    hmms = np.zeros((raw_sequences.shape[0], raw_sequences.shape[1], 50))
    for i, (x, s) in tqdm(enumerate(zip(raw_sequences, dataset.primary))):
        idx = np.where(x > 0)[0]
        hmms[i, idx] = dataset.primary[i][:, :50]

    raw_sequences = np.array([to_onehot(seq, (None, len(AA_IDS) + 1)) for seq in raw_sequences])

    pickle.dump([raw_sequences, None, None, hmms], open(f"{DATA}/{DATASET}/rbm_data.pkl", "wb"))


