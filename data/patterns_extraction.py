import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
from tqdm import tqdm

import biotite

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf
import biotite.structure as struc

from utils import *
from config import *

# SECONDARY STRUCTURE EXTRACTION

def get_structures(pfam_id):
    mapping = pd.read_table(f"{CROSS}/pdb_to_pfam.txt")

    mapping["PFAM_ID"] = mapping.PFAM_ACC.apply(lambda x: x.split(".")[0])
    fam = mapping[mapping.PFAM_ID == pfam_id]

    structfam = set()
    for x in fam.itertuples():
        structfam.add((x.PDB_ID, x.CHAIN_ID, int(x.PdbResNumStart), int(x.PdbResNumEnd)))
    return structfam

def build_patterns(structfam, folder):
    patterns = []
    for pdb, c, start, end in tqdm(structfam):
        file_name = rcsb.fetch(pdb, "mmtf", biotite.temp_dir())
        mmtf_file = mmtf.MMTFFile()
        mmtf_file.read(file_name)

        array = mmtf.get_structure(mmtf_file, model=1)
        tk_dimer = array[struc.filter_amino_acids(array)]

        # The chain ID corresponding to each residue
        chain_id_per_res = array.chain_id[struc.get_residue_starts(tk_dimer)]

        sse = mmtf_file["secStructList"]
        sse = sse[:chain_id_per_res.shape[0]][chain_id_per_res == c]
        sse = np.array(sse[start: end + 1])
        sse = np.array([sec_struct_codes[code % 8] for code in sse], dtype="U1")

        sse8 = to_onehot([dssp_codes[x] for x in sse], (None, 8))
        dss8 = (sse8[1:] - sse8[:-1])
        cls = to_onehot(np.where(dss8 == -1)[1], (None, 8)).T
        bbox = np.array([np.where(dss8 == 1)[0], np.where(dss8 == -1)[0], *cls]).T
        pat8 = np.argmax(bbox[:, 2:], 1)

        sse3 = to_onehot([abc_codes[dssp_to_abc[x]] for x in sse], (None, 3))
        dss3 = (sse3[1:] - sse3[:-1])
        cls = to_onehot(np.where(dss3 == -1)[1], (None, 3)).T
        bbox = np.array([np.where(dss3 == 1)[0], np.where(dss3 == -1)[0], *cls]).T
        pat3 = np.argmax(bbox[:, 2:], 1)
        patterns.append((pat3, pat8))
    if len(patterns) == 0:
        print("No pattern find")
        return None, None, None, None
    c_patterns3, n_patterns3, c_patterns8, n_patterns8, weights = [], [], [], [], []
    for pat3, pat8 in patterns:
        char_pat8 = "".join([sec_struct_codes[x] for x in pat8])
        char_pat3 = "".join(["abc"[x] for x in pat3])
        c_patterns8.append(char_pat8)
        n_patterns8.append(list(pat8))
        c_patterns3.append(char_pat3)
        n_patterns3.append(list(pat3))
    occ_sum8 = dict()
    occ_sum3 = dict()

    correspondings8 = dict()
    correspondings3 = dict()
    for c8, n8, c3, n3 in zip(c_patterns8, n_patterns8, c_patterns3, n_patterns3):
        if c3[0] != "c":
            c3 = "c"+c3
            n3 = [2]+n3
        if c3[-1] != "c":
            c3 = c3+"c"
            n3 = n3+[2]
        if c8[0] != "C":
            c8 = "C"+c8
            n8 = [7]+n8
        if c8[-1] != "C":
            c8 = c8+"C"
            n8 = n8+[7]
        if c8 not in occ_sum8.keys():
            occ_sum8[c8] = 0
            correspondings8[c8] = c8, n8
        occ_sum8[c8] += 1
        if c3 not in occ_sum3.keys():
            occ_sum3[c3] = 0
            correspondings3[c3] = c3, n3
        occ_sum3[c3] += 1

    c_pattern8, n_pattern8 = correspondings8[max(occ_sum8, key=occ_sum8.get)]
    c_pattern3, n_pattern3 = correspondings3[max(occ_sum3, key=occ_sum3.get)]

    push(f"{folder}/data.pt", "pattern", (c_pattern3, n_pattern3, c_pattern8, n_pattern8))

    return c_pattern3, n_pattern3, c_pattern8, n_pattern8, occ_sum3, occ_sum8