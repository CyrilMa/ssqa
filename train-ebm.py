import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F

from pgm.layers import MarkovChainLayer, OneHotLayer
from pgm.data import Seq_SS_Data
from pgm.ebm import EnergyModel
from pgm.nn import ConvNet
from pgm.metrics import hinge_loss, aa_acc, likelihood_loss
from pgm.utils import I

def train():
    start = time.time()
    model.train()
    in_lays, out_lays = ["visible"], ["hidden"]
    edge = model.get_edge("visible", "hidden")
    edge.gauge = edge.gauge.to(device)
    ais = AIS(model, q)

    for epoch in range(6000):
            # Z estimation
        if not epoch%100:
            print(f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss-Z:.3f} || Reg: {mean_reg:.3f} || Acc: {mean_acc:.3f}''', end="\t")
            ais.update(model)
            Z = ais.run(50, 100)
            print()

        mean_loss, mean_reg, mean_acc = 0, 0, 0
        for batch_idx, data in enumerate(train_loader):
            x = data[0].float().permute(0, 2, 1).to(device)
            w = data[1].float().to(device)
            
            # Sampling
            d_0 = {"visible":x}
            d_0, d_f = model.gibbs_sampling(d_0, in_lays, out_lays, k = 10)
            
            # Optimization
            optimizer.zero_grad()
            e_0, e_f = model(d_0), model(d_f)
            reg = l1b_reg(edge)
            loss = msa_mean(e_f-e_0, w) + gamma * reg
            loss.backward()
            optimizer.step()
            
            
            # Metrics
    #         d_0, d_f = model.gibbs_sampling(d_0, in_lays, out_lays, k = 1)
            acc = aa_acc(d_0["visible"].view(*x.size()), d_f["visible"].view(*x.size()))
            ll = msa_mean(model.integrate_likelihood(d_f, "hidden"),w)/31
            mean_loss = (mean_loss*batch_idx + ll.item())/ (batch_idx+1)
            mean_reg = (mean_reg*batch_idx + gamma*reg)/(batch_idx+1)
            mean_acc = (mean_acc*batch_idx + acc)/(batch_idx+1)
            m, s = int(time.time()-start)//60, int(time.time()-start)%60

        print(f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss-Z:.3f} || Reg: {mean_reg:.3f} || Acc: {mean_acc:.3f}''', end="\r")
        
    
def val(epoch):
    mean_loss, mean_reg, mean_acc = 0, 0, 0
    for batch_idx, data in enumerate(val_loader):
        x = data[0].float().permute(0, 2, 1).to(device)
        s = data[1].float().permute(0, 2, 1).to(device)
        
        # Optimization
        loss = hinge_loss(model, x, s)
        acc = aa_acc(s, -model(x))
        del x; del s

        # Metrics
        mean_loss = (mean_loss*batch_idx + loss.item())/ (batch_idx+1)
        mean_acc = (mean_acc*batch_idx + acc)/ (batch_idx+1)
        

        m, s = int(time.time()-start)//60, int(time.time()-start)%60
        print(f'''Val: {epoch} [{int(100*batch_idx/len(val_loader))}%] || Time: {m} min {s} || Loss: {mean_loss:.3f} ''', end="\r")
        
    print(f'''Val: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || Acc: {mean_acc:.3f}           ''')

DATA = "/home/cyril/Documents/These/data/secondary_structure"
batch_size = 32
N, qx, qs = 512, 21, 3

# DATA

train_dataset = Seq_SS_Data(f"{DATA}/secondary_structure_train.json")
train_loader = DataLoader(train_dataset, batch_size = batch_size, 
                          shuffle = True, drop_last=True)

val_dataset = Seq_SS_Data(f"{DATA}/secondary_structure_valid.json")
val_loader = DataLoader(val_dataset, batch_size = batch_size, 
                        shuffle = True, drop_last=True)

# Transition Matrix

t = np.zeros((3, 3))
for seq, length in zip(train_dataset.ss3, train_dataset.length):
    seq = np.argmax(seq, -1)
    x = seq[0]
    y = seq[0]
    for i in range(2, length):
        x, y = y, seq[i]
        t[x,y] += 1
t /= np.sum(t, 1)

# Model

x = OneHotLayer(torch.zeros(qx*N), N = N, q = qx, name = "x")
s = MarkovChainLayer(t, N = N, q = qs, name = "ss")

Dx = ConvNet(qx, qs)
Ds = I
model = EnergyModel(x, s, Dx, Ds)

optimizer = optim.Adam(model.parameters(), lr=0.01)
device = torch.device('cpu')

start = time.time()
model.train()
xlays, ylays = ["x", "ss"], []


for epoch in range(6000):
    train(epoch)
    if not epoch%1:
        val(epoch)

