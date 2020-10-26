import pandas as pd
import pickle
from tqdm import tqdm

from pattern_matching.utils import *

import biotite
import biotite.structure as struc

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf


# SECONDARY STRUCTURE EXTRACTION

def get_structures(data, dataset):
    pdb_uniprot = pd.read_csv(f"{data}/cross/uniprot_pdb.csv", index_col=0)
    mapping = pd.read_table(f"{data}/cross/pdb_to_pfam.txt")
    all_names = [x.split("/")[0] for x in pickle.load(open(f"{data}/{dataset}/index.pkl", "rb"))]

    mapping["PFAM_ID"] = mapping.PFAM_ACC.apply(lambda x: x.split(".")[0])
    fam = mapping[mapping.PFAM_ID == dataset.split("/")[0]]

    structfam = set()
    counts = dict()
    for x in fam.itertuples():
        keys = list(pdb_uniprot[pdb_uniprot.pdb == x.PDB_ID].uni)
        for key in keys:
            if key in all_names:
                structfam.add((key, x.PDB_ID, x.CHAIN_ID))
                if key in counts.keys():
                    counts[key] += 1
                else:
                    counts[key] = 1
    structfam2 = set()
    for x in structfam:
        structfam2.add((*x, 1 / counts[x[0]]))
    return structfam2


INDEL, MISS = 3, 50


def lcs(X, Y):
    m, n = len(X), len(Y)
    L = np.zeros((m + 1, n + 1))

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i, j] = max(1, L[i - 1, j - 1] + 1)
            else:
                L[i, j] = max(0, L[i - 1, j] - INDEL, L[i, j - 1] - INDEL, L[i - 1, j - 1] - MISS)

    # Following code is used to print LCS

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i, j = np.unravel_index(L.argmax(), L.shape)
    posX, posY = [], []
    while i > 0 and j > 0:
        if L[i, j] == 0:
            break
        if X[i - 1] == Y[j - 1]:
            i -= 1;
            j -= 1
            posX.append(i);
            posY.append(j)
            continue
        insert, delete, miss = L[i - 1, j] - INDEL, L[i, j - 1] - INDEL, L[i - 1, j - 1] - MISS
        if insert > delete and insert > miss:
            i -= 1
        elif delete > miss:
            j -= 1
        else:
            i -= 1;
            j -= 1
    posX.sort(), posY.sort()
    return len(posX), posX, posY


def pfam_matching(pfam_seqs, S, uniprot_id):
    best_matching = None
    longest = 0
    pos = None
    for key, val in pfam_seqs.items():
        if key.split("/")[0] == uniprot_id:
            length, posX, posY = lcs(S, val)
            if length > longest:
                best_matching = key, length, posX, posY
                longest = length
    return best_matching


def fetch_PDB(pfam_seqs, pdb, c, uniprot_id):
    file_name = rcsb.fetch(pdb, "mmtf", biotite.temp_dir())
    mmtf_file = mmtf.MMTFFile()
    mmtf_file.read(file_name)
    array = mmtf.get_structure(mmtf_file, model=1)
    # Transketolase homodimer
    tk_dimer = array[struc.filter_amino_acids(array)]
    # Transketolase monomer
    tk_mono = tk_dimer[tk_dimer.chain_id == c]

    # The chain ID corresponding to each residue
    chain_id_per_res = array.chain_id[struc.get_residue_starts(tk_dimer)]
    chain_idx = np.where(chain_id_per_res == c)[0]
    seq = np.array(list(mmtf_file["entityList"][0]["sequence"]))[chain_idx]

    key, length, pos_X, pos_Y = pfam_matching(pfam_seqs, seq, uniprot_id)
    sse = mmtf_file["secStructList"]
    sse = sse[:chain_id_per_res.shape[0]][chain_id_per_res == c]
    idx = np.where(sse == -1)[0]
    while len(idx) > 0:
        sse[idx] = sse[idx - 1]
        idx = np.where(sse == -1)[0]
    sse = np.array([sec_struct_codes[code] for code in sse], dtype="U1")
    sse = np.array([dssp_to_abc[e] for e in sse], dtype="U1")[pos_X]
    return key, pos_X, pos_Y, sse


def build_patterns(pfam_seqs, structfam):
    patterns = dict()
    secondary_structure = dict()
    for uniprot_id, pdb, c, weight in tqdm(structfam):
        try:
            key, pos_X, pos_Y, sse = fetch_PDB(pfam_seqs, pdb, c, uniprot_id)
            ss = to_onehot([abc_codes[x] for x in sse], (None, 3))
            ss = np.pad(ss, ((1, 1), (0, 0)), "constant")
            dss = (ss[1:] - ss[:-1])
            cls = to_onehot(np.where(dss == -1)[1], (None, 3)).T
            bbox = np.array([np.where(dss == 1)[0], np.where(dss == -1)[0], *cls]).T
            struct = np.array([abc_codes[x] for x in sse])
            secondary_number = np.zeros(len(struct), dtype=int)
            for i, x in enumerate(bbox):
                secondary_number[int(x[0]):int(x[1])] = i
            patterns[pdb] = np.argmax(bbox[:, 2:], 1)
            secondary_structure[pdb] = key, struct, secondary_number, np.array(pos_Y), weight
        except:
            ()
    c_patterns, n_patterns = [], []
    for k, pat in patterns.items():
        char_pat = "".join(["abc"[x] for x in pat])
        if len(char_pat) and char_pat not in c_patterns:
            c_patterns.append(char_pat)
            n_patterns.append(list(pat))

    return patterns, secondary_structure, c_patterns, n_patterns