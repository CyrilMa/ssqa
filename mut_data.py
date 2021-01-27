import pandas as pd
import os, re, subprocess
import warnings

from data.mutation_data import run_family, run_dataset

from ss_inference import NetSurfP2

import torch

from ssqa import *

from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from config import *
warnings.filterwarnings("ignore")
PATH = "/home/malbranke/data/mut"

torch.set_num_threads(48)


def build_family(fam, uniprotid):
    print(f"Starting with family {fam} (Uniprot ID : {uniprotid})")
    _, _ = run_family(PATH, fam, uniprotid)
    print("Success")
    return


def best_temperature(X, y):
    e, dpunsup, dpsup, pmunsup_low, pmsup_low, pmunsup_high, pmsup_high = X
    clf = LinearRegression()
    clf.fit(torch.cat([e[:, None], dpunsup[:, None], pmunsup_low[:, None], pmunsup_high[:, None]], 1), y)
    Wunsup = clf.coef_[0], clf.coef_[1], clf.coef_[2], clf.coef_[3]
    clf.fit(torch.cat([e[:, None], dpsup[:, None], pmsup_low[:, None], pmsup_high[:, None]], 1), y)
    Wsup = clf.coef_[0], clf.coef_[1], clf.coef_[2], clf.coef_[3]
    return Wunsup, Wsup


def cv_spearmanr(ssqa, edca, dp, pm, y, N=5):
    cv = KFold(n_splits=N, shuffle=True)
    rho_scores = {"E": 0, "sup/DP": 0, "sup/PM": 0, "sup/PM+DP": 0, "sup/E+DP": 0, "sup/E+PM": 0, "sup/E+DP+PM": 0,
                  "unsup/DP": 0, "unsup/PM": 0, "unsup/PM+DP": 0, "unsup/E+DP": 0, "unsup/E+PM": 0, "unsup/E+DP+PM": 0}
    for i, (train_index, test_index) in enumerate(cv.split(edca)):
        low, high = np.quantile(y[train_index], [0.1])[0], np.quantile(y[train_index], [0.9])[0]
        low_index = np.where(y[train_index] <= low)[0]
        high_index = np.where(y[train_index] >= high)[0]

        ssqa.train(dp[train_index], pm[train_index], y[train_index], low_index)
        _, pmunsup_low, _, pmsup_low = ssqa.predict(dp[test_index], pm[test_index])

        ssqa.train(dp[train_index], pm[train_index], y[train_index], high_index)
        dpunsup, pmunsup_high, dpsup, pmsup_high = ssqa.predict(dp[test_index], pm[test_index])

        e = torch.tensor(edca[test_index])
        dpunsup, dpsup, pmunsup_low, pmsup_low, pmunsup_high, pmsup_high = \
            torch.tensor(dpunsup),torch.tensor(dpsup),torch.tensor(pmunsup_low),torch.tensor(pmsup_low),torch.tensor(pmunsup_high),torch.tensor(pmsup_high)
        y_test = y[test_index]
        (wu_e, wu_dp, wu_pm, wu_pm2), (ws_e, ws_dp, ws_pm, ws_pm2) = best_temperature(
            [e, dpunsup, dpsup, pmunsup_low, pmsup_low, pmunsup_high, pmsup_high], y_test)

        rho_scores["E"] += np.abs(spearmanr(y_test, e)[0]) / N

        pmsup = pmsup_low + ws_pm2/ws_pm * pmsup_high
        pmunsup = pmunsup_low + wu_pm2/wu_pm * pmunsup_high

        rho_scores["sup/DP"] += np.abs(spearmanr(y_test, dpsup)[0]) / N
        rho_scores["sup/PM"] += np.abs(spearmanr(y_test, pmsup)[0]) / N
        rho_scores["sup/PM+DP"] += np.abs(spearmanr(y_test, ws_dp * dpsup + ws_pm * pmsup)[0]) / N
        rho_scores["sup/E+DP"] += np.abs(spearmanr(y_test, ws_e * e + ws_dp * dpsup)[0]) / N
        rho_scores["sup/E+PM"] += np.abs(spearmanr(y_test, ws_e * e + ws_pm * pmsup)[0]) / N
        rho_scores["sup/E+DP+PM"] += np.abs(spearmanr(y_test, ws_e * e + ws_dp * dpsup + ws_pm * pmsup)[0]) / N

        rho_scores["unsup/DP"] += np.abs(spearmanr(y_test, dpunsup)[0]) / N
        rho_scores["unsup/PM"] += np.abs(spearmanr(y_test, pmunsup)[0]) / N
        rho_scores["unsup/PM+DP"] += np.abs(spearmanr(y_test, wu_dp * dpunsup + wu_pm * pmunsup)[0]) / N
        rho_scores["unsup/E+DP"] += np.abs(spearmanr(y_test, wu_e * e + ws_dp * dpunsup)[0]) / N
        rho_scores["unsup/E+PM"] += np.abs(spearmanr(y_test, wu_e * e + ws_pm * pmunsup)[0]) / N
        rho_scores["unsup/E+DP+PM"] += np.abs(spearmanr(y_test, wu_e * e + wu_dp * dpunsup + wu_pm * pmunsup)[0]) / N
    return rho_scores


def build_metrics():
    meta_df = pd.read_excel(f"{PATH}/meta.xlsx", index_col=0)
    rho_df = pd.DataFrame(columns=["fam", "dataset", "exp", "uniprotid", "inpdb", "length", "size", "ind/E",
                                   "ind/sup/DP", "ind/sup/PM", "ind/sup/PM+DP", "ind/sup/E+DP", "ind/sup/E+PM",
                                   "ind/sup/E+DP+PM",
                                   "ind/unsup/DP", "ind/unsup/PM", "ind/unsup/PM+DP", "ind/unsup/E+DP",
                                   "ind/unsup/E+PM", "ind/unsup/E+DP+PM", "dca/E",
                                   "dca/sup/DP", "dca/sup/PM", "dca/sup/PM+DP", "dca/sup/E+DP", "dca/sup/E+PM",
                                   "dca/sup/E+DP+PM",
                                   "dca/unsup/DP", "dca/unsup/PM", "dca/unsup/PM+DP", "dca/unsup/E+DP",
                                   "dca/unsup/E+PM", "dca/unsup/E+DP+PM"
                                   ])
   # rho_df.to_csv(f"{PATH}/rho_df.csv")
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
        try:
            directory = os.listdir(f"{PATH}/{family}")
            if f"{family}.fasta" not in directory or f"{family}_{name_dataset}.csv" not in directory:
                print("Missing files")
            if "data.pt" not in directory:
                build_family(family, uniprotid)
            if f"{name_dataset}_data.pt" not in directory:
                run_dataset(PATH, family, name_dataset)
            dataset = SSQAData_QA(f"{PATH}/{family}/data.pt")
            pattern = dataset.c_pattern3, dataset.n_pattern3, dataset.c_pattern8, dataset.n_pattern8
            dataset = SSQAData_QA(f"{PATH}/{family}/{name_dataset}_data.pt")
            model_ss = NetSurfP2(50, "nsp2")
            model_ss = model_ss.to("cuda")
            model_ss.load_state_dict(torch.load(f"{UTILS}/nsp_50feats.h5"))

            seq_hmm = dataset.seq_hmm
            size = seq_hmm.size(-1)

            SS_HMM3 = torch.ones(3, size) / 3
            SS_HMM8 = torch.ones(8, size) / 8
            ss_hmm = torch.tensor(dataset[0]).float()
            active_idx = torch.where((ss_hmm[:20].sum(0) > 0))[0]
            pred = model_ss(ss_hmm[None, :, active_idx].cuda())
            SS_HMM3[:, active_idx] = F.softmax(pred[2][0], 0).cpu()
            SS_HMM8[:, active_idx] = F.softmax(pred[1][0], 0).cpu()
            SS_HMM3 = SS_HMM3[None]
            SS_HMM8 = SS_HMM8[None]
            X = torch.cat([data[None] for data in dataset], 0)
            ssqa = SSQAMut(model_ss, pattern, seq_hmm, SS_HMM3, SS_HMM8)

            dp, pm = ssqa.featuring(X)
            mut_df = pd.read_csv(f"{MUT_DATA}/{family}/{name_dataset}_mutation_sequences.csv", index_col=0)
            isna = (mut_df["effect_prediction_epistatic"].isna()) | (mut_df["effect_prediction_independent"].isna())
        except:
            continue
        for exp in exp_columns:
            isnaexp = (mut_df[exp].isna()) | isna
            y = mut_df[~isnaexp][exp].values
            edca = torch.tensor(mut_df["effect_prediction_epistatic"][~isnaexp]).float()
            eind = torch.tensor(mut_df["effect_prediction_independent"][~isnaexp]).float()
            rho_scores_ind = cv_spearmanr(ssqa, eind, dp, pm, y)
            rho_scores_dca = cv_spearmanr(ssqa, edca, dp, pm, y)

            print("")
            print(f"Size : {len(y)}")
            print(f"Length : {size}")
            for k, v in rho_scores_dca.items():
                print(f"{k} : {v:.3f}")
            entry = [family, name_dataset, exp, uniprotid, in_pdb, size, len(y)]
            entry += list(rho_scores_ind.values())
            entry += list(rho_scores_dca.values())

            rho_df = pd.read_csv(f"{PATH}/rho_df.csv", index_col=0)
            rho_df.loc[f"{name_dataset}_{exp}"] = entry
            rho_df.to_csv(f"{PATH}/rho_df.csv")
            print("")
    print("---------------------------")


build_metrics()