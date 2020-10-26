import pandas as pd
import pickle
from tqdm import tqdm
import subprocess

from random import shuffle
from pattern_matching.utils import *

import biotite
import biotite.structure as struc

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import netsurfp2 as nsp2


# HMM PROFILES

