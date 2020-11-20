import pandas as pd
import os, re, subprocess
import warnings
import numpy as np
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data import build_profiles
from data.mutation_data import run_family, run_dataset

from ss_inference.data import SecondaryStructureRawDataset, collate_sequences
from ss_inference.model import NetSurfP2

from pattern.pattern import Matching, PatternMatching, PatternWithoutMatching
from pattern.inference import PatternMatchingInference

from scipy.stats import spearmanr, rankdata
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from config import *
warnings.filterwarnings("ignore")
PATH = "/home/malbranke/mutation_data"

def build_family(fam, uniprotid):
    print(f"Starting with family {fam} (Uniprot ID : {uniprotid})")
    pattern, ratio_covered = run_family(PATH, fam, uniprotid)
    print("Success")
    return

def cv_spearmanr(X, y, N = 5):
    cv = KFold(n_splits=N, shuffle=True)
    pred = np.zeros(len(y))
    all_ = 0
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        clf = RandomForestRegressor(50)
        #clf = SVR()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        pred[test_index] = clf.predict(X_test)
        all_ += np.abs(spearmanr(y_test, clf.predict(X_test))[0])
    return all_/N

def interpolate(X, V, y, N=5):
    cv = KFold(n_splits=N, shuffle=True)
    all_ = 0
    t_ = []
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        clf1 = RandomForestRegressor(50)
        clf2 = RandomForestRegressor(50)

        #clf1 = SVR()
        #clf2 = SVR()

        X_train, X_test = X[train_index], X[test_index]
        V_train, V_test = V[train_index], V[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf1.fit(X_train, y_train)
        pred_X = clf1.predict(X_train)
        clf2.fit(V_train, y_train)
        pred_V = clf2.predict(V_train)
        spears = np.abs(np.array([spearmanr(y_train, t*pred_X + (1-t)*pred_V)[0] for t in np.arange(0,1.01,0.01)]))
        t_.append(spears.argmax()/100)
        t = t_[-1]
        pred_X = clf1.predict(X_test)
        pred_V = clf2.predict(V_test)
        all_ += spearmanr(y_test, t*pred_X + (1-t)*pred_V)[0]
    return all_/N, np.median(np.array(t_))

def build_metrics():
    meta_df = pd.read_excel(f"{PATH}/meta.xlsx", index_col=0)

    for metadata in meta_df.itertuples():
        family = metadata.family
        name_dataset = metadata.dataset
        uniprotid = metadata.uniprot
        in_pdb = metadata.in_PDB
        exp_columns = re.findall(r"\w\w*", metadata.exp_columns)

        print(f"Family : {family}")
        print(f"Dataset : {name_dataset}")
        print(f"Uniprot ID : {uniprotid}")
        print(f"Experimental Columns : {exp_columns}")
        print()
        directory = os.listdir(f"{PATH}/{family}")
        if f"{family}.fasta" not in directory or f"{family}_{name_dataset}.csv" not in directory:
            print("Missing files")
        if "aligned.hhm" not in directory or "patterns.pkl" not in directory:
            build_family(family, uniprotid)
        if f"{name_dataset}_hmm.pkl" not in directory:
            run_dataset(PATH, family, name_dataset)
        try:

            dataset = SecondaryStructureRawDataset(f"{PATH}/{family}/hmm.pkl")
            model_ss3 = NetSurfP2(50, "nsp2")
            model_ss3 = model_ss3.to("cuda")
            model_ss3.load_state_dict(torch.load(f"{DATA}/secondary_structure/lstm_50feats.h5"))
            x = torch.tensor(dataset[0][0])[None].permute(0, 2, 1).float().cuda()
            ss_hmm = model_ss3(x)[2][0].detach().cpu()
            seq_hmm = torch.tensor(dataset[0][0]).t()[20:]
            size = seq_hmm.size(-1)

            n_pattern, c_pattern, m_pat, M_pat = pickle.load(open(f"{PATH}/{family}/patterns.pkl", "rb"))
            Q = np.ones((3, size + 1, size + 1)) * (-np.inf)
            for i in range(size + 1):
                Q[:3, i, i + 1:] = 0
            Q = Q.reshape(1, *Q.shape)
            regex = ([(i, None, None) for i in n_pattern])


            dataset = SecondaryStructureRawDataset(f"{PATH}/{family}/{name_dataset}_hmm.pkl")
            loader = DataLoader(dataset, batch_size=16,
                                shuffle=False, drop_last=False, collate_fn=collate_sequences)


            matcher1 = PatternMatching(model_ss3, pattern = regex, Q=Q,
                                      seq_hmm=seq_hmm, ss_hmm=None,
                                      size=size, name= c_pattern)
            matcher2 = PatternWithoutMatching(model_ss3, pattern=regex, Q=Q,
                                             seq_hmm=seq_hmm, ss_hmm=ss_hmm,
                                             size=size, name=c_pattern)

            ls, L, V = [], [], []
            for batch_idx, data in tqdm(enumerate(loader)):
                x = data[0].permute(0, 2, 1).float()
                m = Matching(x)
                matcher1(m)
                v = matcher2(m)
                L.append(m.L), ls.append(m.ls), V.append(v)
                del m
                torch.cuda.empty_cache()
            ls, V, L = torch.cat(ls, 0), torch.cat(V, 0), torch.cat(L, 0)
            L = L.clamp(1e-8,1)
            mut_df = pd.read_csv(f"{PATH}/{family}/{name_dataset}_mutation_sequences.csv", index_col=0)
            isna = (mut_df["effect_prediction_epistatic"].isna()) | (mut_df["effect_prediction_independent"].isna())
            if len(mut_df[~isna]) == 0:
                continue
            u = torch.arange(30).view(1,1,-1)
            X = (L*u).mean(-1)
        except:
            continue

        for exp in exp_columns:
            try:
                isnaexp = (mut_df[exp].isna()) | isna
                y = mut_df[~isnaexp][exp].values
                high_bound, low_bound = np.quantile(y, 0.8), np.quantile(y, 0.2)

                # Dot build
                V_ = V[~isnaexp]
                V_ = V_[:, torch.where(V_.std(0) != 0)[0]]

                m, s = V_[mut_df[~isnaexp][exp] >= high_bound].mean(0)[None], \
                       V_[mut_df[~isnaexp][exp] >= high_bound].std(0)[None]
                dot_high = (V_ - m) / s
                m, s = V_[mut_df[~isnaexp][exp] <= low_bound].mean(0)[None], \
                       V_[mut_df[~isnaexp][exp] <= low_bound].std(0)[None]
                dot_low = (V_ - m) / s

                dot = torch.cat([dot_low, dot_high], 1)
                dot = dot[:, torch.where((dot.sum(0) == dot.sum(0)) & (dot.std(0) != 0))[0]]
                dot_ind = torch.cat([dot, torch.tensor(mut_df[~isnaexp]["effect_prediction_independent"].values)[:, None]], 1)
                dot_epi = torch.cat([dot, torch.tensor(mut_df[~isnaexp]["effect_prediction_epistatic"].values)[:, None]], 1)

                dot = torch.cat([dot] + [torch.tensor(rankdata(d, method='ordinal')).view(-1, 1) for d in dot.t()], 1)
                dot_ind = torch.cat([dot_ind] + [torch.tensor(rankdata(d, method='ordinal')).view(-1, 1) for d in dot_ind.t()],1)
                dot_epi = torch.cat([dot_epi] + [torch.tensor(rankdata(d, method='ordinal')).view(-1, 1) for d in dot_epi.t()], 1)

                dot = (dot - dot.mean(0)[None]) / dot.std(0)[None]
                dot_ind = (dot_ind - dot_ind.mean(0)[None]) / dot_ind.std(0)[None]
                dot_epi = (dot_epi - dot_epi.mean(0)[None]) / dot_epi.std(0)[None]

                rho_dot = cv_spearmanr(dot, y)
                rho_dot_ind = cv_spearmanr(dot_ind, y)
                rho_dot_epi = cv_spearmanr(dot_epi, y)

                # Matching Expect build
                X_ = X[~isnaexp]
                m,s  = X_[mut_df[~isnaexp][exp] >= high_bound].mean(0)[None], X_[mut_df[~isnaexp][exp] >= high_bound].std(0)[None]
                X_high = (X_-m)/s
                m,s = X_[mut_df[~isnaexp][exp] <= low_bound].mean(0)[None], X_[mut_df[~isnaexp][exp] <= low_bound].std(0)[None]
                X_low = (X_-m)/s
                matching_expect = torch.cat([X_low, X_high], 1)
                matching_expect_ind = torch.cat([matching_expect, torch.tensor(mut_df[~isnaexp]["effect_prediction_independent"].values)[:,None]], 1)
                matching_expect_epi = torch.cat([matching_expect, torch.tensor(mut_df[~isnaexp]["effect_prediction_epistatic"].values)[:,None]], 1)

                matching_expect = torch.cat([matching_expect] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_expect.t()], 1)
                matching_expect_ind = torch.cat([matching_expect_ind] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_expect_ind.t()], 1)
                matching_expect_epi = torch.cat([matching_expect_epi] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_expect_epi.t()], 1)

                matching_expect = (matching_expect - matching_expect.mean(0)[None])/matching_expect.std(0)[None]
                matching_expect_ind = (matching_expect_ind - matching_expect_ind.mean(0)[None])/matching_expect_ind.std(0)[None]
                matching_expect_epi = (matching_expect_epi - matching_expect_epi.mean(0)[None])/matching_expect_epi.std(0)[None]

                rho_matching_expect = cv_spearmanr(matching_expect, y)
                rho_matching_expect_ind = cv_spearmanr(matching_expect_ind, y)
                rho_matching_expect_epi = cv_spearmanr(matching_expect_epi, y)

                # Matching Divergence build
                L_0 = L[~isnaexp][mut_df[~isnaexp][exp] >= high_bound].mean(0)
                div_high = (L_0 * (torch.log(L_0 + 1e-8) - torch.log(L[~isnaexp]+ 1e-8))).sum(-1)
                L_0 = L[~isnaexp][mut_df[~isnaexp][exp] <= low_bound].mean(0)
                div_low = (L_0 * (torch.log(L_0 + 1e-8) - torch.log(L[~isnaexp]+ 1e-8))).sum(-1)

                matching_div = torch.cat([div_low, div_high], 1)
                matching_div_ind = torch.cat([matching_div, torch.tensor(mut_df[~isnaexp]["effect_prediction_independent"].values)[:,None]], 1)
                matching_div_epi = torch.cat([matching_div, torch.tensor(mut_df[~isnaexp]["effect_prediction_epistatic"].values)[:,None]], 1)

                matching_div = torch.cat([matching_div] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_div.t()], 1)
                matching_div_ind = torch.cat([matching_div_ind] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_div_ind.t()], 1)
                matching_div_epi = torch.cat([matching_div_epi] + [torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_div_epi.t()], 1)

                matching_div = (matching_div - matching_div.mean(0)[None])/matching_div.std(0)[None]
                matching_div_ind = (matching_div_ind - matching_div_ind.mean(0)[None])/matching_div_ind.std(0)[None]
                matching_div_epi = (matching_div_epi - matching_div_epi.mean(0)[None])/matching_div_epi.std(0)[None]

                rho_matching_div = cv_spearmanr(matching_div, y)
                rho_matching_div_ind = cv_spearmanr(matching_div_ind, y)
                rho_matching_div_epi = cv_spearmanr(matching_div_epi, y)

                # Combine
                if rho_matching_div_epi > rho_matching_expect_epi:
                    rho_matching = rho_matching_div
                    rho_matching_ind = rho_matching_div_ind
                    rho_matching_epi = rho_matching_div_epi
                    combine_epi = torch.cat([matching_div, dot_epi], 1)
                    rho_interpolate_epi, t = interpolate(matching_div_epi, dot_epi, y)
                else:
                    rho_matching = rho_matching_expect
                    rho_matching_ind = rho_matching_expect_ind
                    rho_matching_epi = rho_matching_expect_epi
                    combine_epi = torch.cat([matching_expect, dot_epi], 1)
                    rho_interpolate_epi, t = interpolate(matching_expect_epi, dot_epi, y)
                rho_combine_epi = cv_spearmanr(combine_epi, y)

                rho_independent = spearmanr(y, mut_df[~isnaexp]["effect_prediction_independent"].values)[0]
                rho_epistatic = spearmanr(y, mut_df[~isnaexp]["effect_prediction_epistatic"].values)[0]
                entry = [family, name_dataset, exp, in_pdb, len(y), len(ss_hmm[0]), rho_independent, rho_epistatic,
                         rho_dot, rho_dot_ind, rho_dot_epi,
                         rho_matching, rho_matching_ind, rho_matching_epi,
                         rho_interpolate_epi, t, rho_combine_epi]

                print("")
                print(f"Size : {len(y)}")
                print(f"Length : {len(ss_hmm[0])}")
                print(f"Independent | Rho = {rho_independent:.3f}")
                print(f"Epistatic | Rho = {rho_epistatic:.3f}")
                print(f"Dot | Rho = {rho_dot:.3f}")
                print(f"Dot + Independent | Rho = {rho_dot_ind:.3f}")
                print(f"Dot + Epistatic | Rho = {rho_dot_epi:.3f}")
                print(f"Matching | Rho = {rho_matching:.3f}")
                print(f"Matching + Independent | Rho = {rho_matching_ind:.3f}")
                print(f"Matching + Epistatic | Rho = {rho_matching_epi:.3f}")
                print(f"Interpolation | Rho = {rho_interpolate_epi:.3f}, t = {t}")
                print(f"Combination Epistatic | Rho = {rho_combine_epi:.3f}")

                rho_df = pd.read_csv(f"{PATH}/rho_df_randomforest.csv", index_col = 0)
                rho_df.loc[f"{name_dataset}_{exp}"] = entry
                rho_df.to_csv(f"{PATH}/rho_df_randomforest.csv")
                print("")
            except:
                continue
        print("---------------------------")

def build_metrics2():
    meta_df = pd.read_excel(f"{PATH}/meta.xlsx", index_col=0)

    for metadata in meta_df.itertuples():
        try:
            family = metadata.family
            name_dataset = metadata.dataset
            uniprotid = metadata.uniprot
            in_pdb = metadata.in_PDB
            exp_columns = re.findall(r"\w\w*", metadata.exp_columns)

            print(f"Family : {family}")
            print(f"Dataset : {name_dataset}")
            print(f"Uniprot ID : {uniprotid}")
            print(f"Experimental Columns : {exp_columns}")
            print()
            directory = os.listdir(f"{PATH}/{family}")
            if f"{family}.fasta" not in directory or f"{family}_{name_dataset}.csv" not in directory:
                print("Missing files")
            if "aligned.hhm" not in directory or "patterns.pkl" not in directory:
                build_family(family, uniprotid)
            if f"{name_dataset}_hmm.pkl" not in directory:
                run_dataset(PATH, family, name_dataset)

            dataset = SecondaryStructureRawDataset(f"{PATH}/{family}/hmm.pkl")
            model_ss3 = NetSurfP2(50, "nsp2")
            model_ss3 = model_ss3.to("cuda")
            model_ss3.load_state_dict(torch.load(f"{DATA}/secondary_structure/lstm_50feats.h5"))
            x_0 = torch.tensor(dataset[0][0])[None].permute(0, 2, 1).float().cuda()
            ss_hmm = model_ss3(x_0)[2][0].detach().cpu()
            seq_hmm = torch.tensor(dataset[0][0]).t()[20:]
            size = seq_hmm.size(-1)

            n_pattern, c_pattern, m_pat, M_pat = pickle.load(open(f"{PATH}/{family}/patterns.pkl", "rb"))
            Q = np.ones((3, size + 1, size + 1)) * (-np.inf)
            for i in range(size + 1):
                Q[:3, i, i + 1:] = 0
            Q = Q.reshape(1, *Q.shape)
            regex = ([(i, None, None) for i in n_pattern])


            dataset = SecondaryStructureRawDataset(f"{PATH}/{family}/{name_dataset}_hmm.pkl")
            loader = DataLoader(dataset, batch_size=16,
                                shuffle=False, drop_last=False, collate_fn=collate_sequences)


            matcher1 = PatternMatching(model_ss3, pattern = regex, Q=Q,
                                      seq_hmm=seq_hmm, ss_hmm=None,
                                      size=size, name= c_pattern)
            matcher2 = PatternWithoutMatching(model_ss3, pattern=regex, Q=Q,
                                             seq_hmm=seq_hmm, ss_hmm=ss_hmm,
                                             size=size, name=c_pattern)

            m = Matching(x_0)
            matcher1(m)
            L_0 = m.L

            ls, L, V = [], [], []
            for batch_idx, data in tqdm(enumerate(loader)):
                x = data[0].permute(0, 2, 1).float()
                m = Matching(x)
                matcher1(m)
                v = matcher2(m)
                L.append(m.L), ls.append(m.ls), V.append(v)
                del m
                torch.cuda.empty_cache()
            ls, V, L = torch.cat(ls, 0), torch.cat(V, 0), torch.cat(L, 0)
            L = L.clamp(1e-8,1)
            mut_df = pd.read_csv(f"{PATH}/{family}/{name_dataset}_mutation_sequences.csv", index_col=0)
            isna = (mut_df["effect_prediction_epistatic"].isna()) | (mut_df["effect_prediction_independent"].isna())
            if len(mut_df[~isna]) == 0:
                continue
            u = torch.arange(30).view(1,1,-1)
            X = (L*u).mean(-1)
            X_0 = (L_0*u).mean(-1)

        except:
            continue

        for exp in exp_columns:
            try:
                isnaexp = (mut_df[exp].isna()) | isna
                y = mut_df[~isnaexp][exp].values

                # Dot build
                V_ = V[~isnaexp]
                dot = V_
                dot = dot[:, torch.where((dot.sum(0) == dot.sum(0)) & (dot.std(0) != 0))[0]]
                dot_ind = torch.cat([dot, torch.tensor(mut_df[~isnaexp]["effect_prediction_independent"].values)[:, None]], 1)
                dot_epi = torch.cat([dot, torch.tensor(mut_df[~isnaexp]["effect_prediction_epistatic"].values)[:, None]], 1)

                dot = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1, 1) for d in dot.t()], 1).max(1)[0].float()
                dot_ind = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1, 1) for d in dot_ind.t()],1).max(1)[0].float()
                dot_epi = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1, 1) for d in dot_epi.t()], 1).max(1)[0].float()

                dot = (dot - dot.mean(0)[None]) / dot.std(0)[None]
                dot_ind = (dot_ind - dot_ind.mean(0)[None]) / dot_ind.std(0)[None]
                dot_epi = (dot_epi - dot_epi.mean(0)[None]) / dot_epi.std(0)[None]

                rho_dot = spearmanr(y, dot)[0]
                rho_dot_ind = spearmanr(y, dot_ind)[0]
                rho_dot_epi = spearmanr(y, dot_epi)[0]

                # Matching Expect build
                X_ = X[~isnaexp]
                matching_expect = (X_ - X_0)/X_0
                matching_expect_ind = torch.cat([matching_expect, torch.tensor(mut_df[~isnaexp]["effect_prediction_independent"].values)[:,None]], 1)
                matching_expect_epi = torch.cat([matching_expect, torch.tensor(mut_df[~isnaexp]["effect_prediction_epistatic"].values)[:,None]], 1)

                matching_expect = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_expect.t()], 1).max(1)[0].float()
                matching_expect_ind = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_expect_ind.t()], 1).max(1)[0].float()
                matching_expect_epi = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_expect_epi.t()], 1).max(1)[0].float()

                matching_expect = (matching_expect - matching_expect.mean(0)[None])/matching_expect.std(0)[None]
                matching_expect_ind = (matching_expect_ind - matching_expect_ind.mean(0)[None])/matching_expect_ind.std(0)[None]
                matching_expect_epi = (matching_expect_epi - matching_expect_epi.mean(0)[None])/matching_expect_epi.std(0)[None]

                rho_matching_expect = spearmanr(y,matching_expect)[0]
                rho_matching_expect_ind = spearmanr(y,matching_expect_ind)[0]
                rho_matching_expect_epi = spearmanr(y,matching_expect_epi)[0]

                # Matching Divergence build
                L_0_ = L[~isnaexp].mean(0)
                div_high = (L_0_ * (torch.log(L_0 + 1e-8) - torch.log(L[~isnaexp]+ 1e-8))).sum(-1)

                matching_div = div_high
                matching_div_ind = torch.cat([matching_div, torch.tensor(mut_df[~isnaexp]["effect_prediction_independent"].values)[:,None]], 1)
                matching_div_epi = torch.cat([matching_div, torch.tensor(mut_df[~isnaexp]["effect_prediction_epistatic"].values)[:,None]], 1)

                matching_div = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_div.t()], 1).max(1)[0].float()
                matching_div_ind = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_div_ind.t()], 1).max(1)[0].float()
                matching_div_epi = torch.cat([torch.tensor(rankdata(d, method='ordinal')).view(-1,1) for d in matching_div_epi.t()], 1).max(1)[0].float()

                matching_div = (matching_div - matching_div.mean(0)[None])/matching_div.std(0)[None]
                matching_div_ind = (matching_div_ind - matching_div_ind.mean(0)[None])/matching_div_ind.std(0)[None]
                matching_div_epi = (matching_div_epi - matching_div_epi.mean(0)[None])/matching_div_epi.std(0)[None]

                rho_matching_div = spearmanr(y,matching_div)[0]
                rho_matching_div_ind = spearmanr(y,matching_div_ind)[0]
                rho_matching_div_epi = spearmanr(y,matching_div_epi)[0]

                # Combine
                if rho_matching_div_epi > rho_matching_expect_epi:
                    rho_matching = rho_matching_div
                    rho_matching_ind = rho_matching_div_ind
                    rho_matching_epi = rho_matching_div_epi
                    combine_epi = torch.cat([matching_div[:,None], dot_epi[:,None]], 1).max(1)[0]
                else:
                    rho_matching = rho_matching_expect
                    rho_matching_ind = rho_matching_expect_ind
                    rho_matching_epi = rho_matching_expect_epi
                    combine_epi = torch.cat([matching_expect[:,None], dot_epi[:,None]], 1).max(1)[0]
                rho_combine_epi = spearmanr(y, combine_epi)[0]

                rho_independent = spearmanr(y, mut_df[~isnaexp]["effect_prediction_independent"].values)[0]
                rho_epistatic = spearmanr(y, mut_df[~isnaexp]["effect_prediction_epistatic"].values)[0]
                entry = [family, name_dataset, exp, in_pdb, len(y), len(ss_hmm[0]), rho_independent, rho_epistatic,
                         rho_dot, rho_dot_ind, rho_dot_epi,
                         rho_matching, rho_matching_ind, rho_matching_epi,
                         None, None, rho_combine_epi]

                print("")
                print(f"Size : {len(y)}")
                print(f"Length : {len(ss_hmm[0])}")
                print(f"Independent | Rho = {rho_independent:.3f}")
                print(f"Epistatic | Rho = {rho_epistatic:.3f}")
                print(f"Dot | Rho = {rho_dot:.3f}")
                print(f"Dot + Independent | Rho = {rho_dot_ind:.3f}")
                print(f"Dot + Epistatic | Rho = {rho_dot_epi:.3f}")
                print(f"Matching | Rho = {rho_matching:.3f}")
                print(f"Matching + Independent | Rho = {rho_matching_ind:.3f}")
                print(f"Matching + Epistatic | Rho = {rho_matching_epi:.3f}")
                print(f"Combination Epistatic | Rho = {rho_combine_epi:.3f}")

                rho_df = pd.read_csv(f"{PATH}/rho_df_unsupervised.csv", index_col = 0)
                rho_df.loc[f"{name_dataset}_{exp}"] = entry
                rho_df.to_csv(f"{PATH}/rho_df_unsupervised.csv")
                print("")
            except:
                continue
        print("---------------------------")


build_metrics()