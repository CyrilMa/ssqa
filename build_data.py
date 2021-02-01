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

pfam_data(f"{DATA}/{DATASET}", "full.fasta")
structfam = get_structures(DATASET)
build_patterns(structfam, f"{DATA}/{DATASET}")


