import pandas as pd
import os, re, subprocess
import warnings
import numpy as np
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data_extraction import build_profiles
from data_extraction.mutation_data import run_family, run_dataset

from ss_inference.data import SecondaryStructureRawDataset, collate_sequences
from ss_inference.model import NetSurfP2

from pattern_matching.loss import Matching, PatternMatching
from pattern_matching.inference import PatternMatchingInference

from scipy.stats import spearmanr, rankdata
from sklearn.svm import SVR
from sklearn.model_selection import KFold

from config import *
warnings.filterwarnings("ignore")
PATH = "/home/malbranke/mutation_data"

def build_family(fam, uniprotid):
    print(f"Starting with family {fam} (Uniprot ID : {uniprotid})")
    pattern, ratio_covered = run_family(PATH, fam, uniprotid)
    print("Success")
    if pattern:
        print(f"Pattern Found ! (Ratio : {int(100*ratio_covered)}%")
        print("---------------------")
        return
    print(f"No Pattern Find ! (Ratio : {int(100*ratio_covered)}%")
    print("Beginning Pattern Matching Inference")
    build_profiles(f"{PATH}/{fam}", idxs=[0])
    dataset = SecondaryStructureRawDataset(f"{PATH}/{fam}/hmm.pkl")
    Q = torch.load("Q.pt").float()
    pi = torch.load("pi.pt")[:, 0].float()

    seq_hmm = torch.tensor(dataset[0][0]).t()[20:]
    torch.save(seq_hmm, f"{PATH}/{fam}/hmm.pt")
    _, size = seq_hmm.size()
    torch.cuda.empty_cache()

    model_ss3 = NetSurfP2(50, "nsp2")
    model_ss3 = model_ss3.to("cuda")
    model_ss3.load_state_dict(torch.load(f"{DATA}/secondary_structure/lstm_50feats.h5"))

    inferer = PatternMatchingInference(model_ss3, Q=Q, pi=pi,
                                       seq_hmm=seq_hmm, size=size)
    print("Inference Model build")
    x = torch.tensor(dataset[0][0])[None].permute(0, 2, 1).float()
    m = Matching(x)
    p = inferer(m, 1)[0]

    p_ = p[:, 0]
    idx = torch.where(p_ < 3)[0]
    n_pattern = list(p_[idx].numpy())
    c_pattern = "".join("abc"[x] for x in p_[idx])
    print(f"Pattern Infered : {c_pattern}")
    pickle.dump((n_pattern, c_pattern, 0, x.size(-1)-1), open(f"{PATH}/{fam}/patterns.pkl", "wb"))
    return

def cv_spearmanr(X, y, N = 5):
    cv = KFold(n_splits=N, shuffle=True)
    pred = np.zeros(len(y))
    all_ = 0
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        clf = SVR()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        pred[test_index] = clf.predict(X_test)
        all_ += spearmanr(y_test, clf.predict(X_test))[0]
    return all_/N


def build_metrics():
    meta_df = pd.read_excel(f"{PATH}/meta.xlsx", index_col=0)

    for metadata in meta_df.itertuples():
        try:
            family = metadata.family
            name_dataset = metadata.dataset
            uniprotid = metadata.uniprot
            exp_columns = re.findall(r"\w\w*", metadata.exp_columns)

            print(f"Family : {family}")
            print(f"Dataset : {name_dataset}")
            print(f"Uniprot ID : {uniprotid}")
            print(f"Experimental Columns : {exp_columns}")

            directory = os.listdir(f"{PATH}/{family}")
            if f"{family}.fasta" not in directory or f"{family}_{name_dataset}.csv" not in directory:
                print("Missing files")
            if "aligned.hhm" not in directory or "patterns.pkl" not in directory:
                build_family(family, uniprotid)
            if f"{name_dataset}_hmm.pkl" not in directory:
                run_dataset(PATH, family, name_dataset)

            dataset = SecondaryStructureRawDataset(f"{PATH}/{family}/{name_dataset}_hmm.pkl")
            loader = DataLoader(dataset, batch_size=50,
                                shuffle=False, drop_last=False, collate_fn=collate_sequences)
            model_ss3 = NetSurfP2(50, "nsp2")
            model_ss3 = model_ss3.to("cuda")
            model_ss3.load_state_dict(torch.load(f"{DATA}/secondary_structure/lstm_50feats.h5"))

            seq_hmm = torch.tensor(dataset[0][0]).t()[20:]
            torch.save(seq_hmm, f"{PATH}/{family}/hmm.pt")
            _, size = seq_hmm.size()

            n_pattern, c_pattern, m_pat, M_pat = pickle.load(open(f"{PATH}/{family}/patterns.pkl", "rb"))
            Q = np.ones((3, size + 1, size + 1)) * (-np.inf)
            for i in range(size + 1):
                Q[:3, i, i + 1:] = 0
            Q = Q.reshape(1, *Q.shape)
            regex = ([(i, None, None) for i in n_pattern])

            matcher = PatternMatching(model_ss3, pattern = regex, Q=Q,
                                      seq_hmm=seq_hmm, ss_hmm=None,
                                      size=size, name= c_pattern)

            ls, M, L = [], [], []
            for batch_idx, data in tqdm(enumerate(loader)):
                x = data[0].permute(0, 2, 1).float()
                m = Matching(x)
                matcher(m)
                L.append(m.L), ls.append(m.ls), M.append(m.M)
                del m
                torch.cuda.empty_cache()
            ls, M, L = torch.cat(ls, 0), torch.cat(M, 0), torch.cat(L, 0)
            L = L.clamp(1e-8,1)
            mut_df = pd.read_csv(f"{PATH}/{family}/{name_dataset}_mutation_sequences.csv", index_col=0)
            isna = (mut_df["effect_prediction_epistatic"].isna()) | (mut_df["effect_prediction_independent"].isna())
            if len(mut_df[~isna]) == 0:
                continue
            for exp in exp_columns:
                rho = [family, name_dataset, exp]
                mut_df[f"{exp}_L_high"], mut_df[f"{exp}_L_low"] = None, None
                isnaexp = (mut_df[exp].isna()) | isna
                y = mut_df[~isnaexp][exp].values
                high_bound, low_bound = np.quantile(y, 0.8), np.quantile(y, 0.2)
                L_0 = L[~isnaexp][mut_df[~isnaexp][exp] >= high_bound].mean(0)
                div_high = (L_0 * (torch.log(L_0 + 1e-8) - torch.log(L[~isnaexp]+ 1e-8))).sum(-1)
                L_0 = L[~isnaexp][mut_df[~isnaexp][exp] <= low_bound].mean(0)
                div_low = (L_0 * (torch.log(L_0 + 1e-8) - torch.log(L[~isnaexp]+ 1e-8))).sum(-1)

                div = torch.cat([div_low, div_high], 1)
                div_ind = torch.cat([div, torch.tensor(mut_df[~isnaexp]["effect_prediction_independent"].values)[:,None]], 1)
                div_epi = torch.cat([div, torch.tensor(mut_df[~isnaexp]["effect_prediction_epistatic"].values)[:,None]], 1)

                div = torch.cat([div] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in div.t()], 1)
                div_ind = torch.cat([div_ind] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in div_ind.t()], 1)
                div_epi = torch.cat([div_epi] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in div_epi.t()], 1)

                rho.append(spearmanr(y, mut_df[~isnaexp]["effect_prediction_independent"].values)[0])
                rho.append(spearmanr(y, mut_df[~isnaexp]["effect_prediction_epistatic"].values)[0])
                rho.append(cv_spearmanr(div, y))
                rho.append(cv_spearmanr(div_ind, y))
                rho.append(cv_spearmanr(div_epi, y))

                print()
                print(f"Independent | Rho = {rho[3]}")
                print(f"Epistatic | Rho = {rho[4]}")
                print(f"SSQA | Rho = {rho[5]}")
                print(f"SSQA + Independent | Rho = {rho[6]}")
                print(f"SSQA + Epistatic | Rho = {rho[7]}")



                rho_df = pd.read_csv(f"{PATH}/rho_df.csv", index_col = 0)
                rho_df.loc[f"{name_dataset}_{exp}"] = rho
                rho_df.to_csv(f"{PATH}/rho_df.csv")
            print("-------------------------------------------------------")
        except:
            continue
      #  subprocess.run(f'rm -rf "{PATH}/{family}/{name_dataset}_mutation_sequences.csv"')
       # subprocess.run(f'rm -rf "{PATH}/{family}/{name_dataset}_hmm.pkl"')


build_metrics()