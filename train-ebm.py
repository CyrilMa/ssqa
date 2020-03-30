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

def train(epoch):
    mean_loss, mean_reg, mean_acc = 0, 0, 0
    for batch_idx, data in enumerate(train_loader):
        x = data[0].float().permute(0, 2, 1).to(device)
        s = data[1].float().permute(0, 2, 1).to(device)
        length = data[2].int().to(device)
        
        # Optimization
        optimizer.zero_grad()
        loss = hinge_loss(model, x, s)
        loss.backward()
        optimizer.step()
#         print(d_0["visible"].argmax(-1)[0], d_f["visible"].argmax(-1)[0])
        acc = aa_acc(s, -model(x))

        del x; del s
        # Metrics
        mean_loss = (mean_loss*batch_idx + loss.item())/ (batch_idx+1)
        mean_acc = (mean_acc*batch_idx + acc)/ (batch_idx+1)
        m, s = int(time.time()-start)//60, int(time.time()-start)%60
        print(f'''Train Epoch: {epoch} [{int(100*batch_idx/len(train_loader))}%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || Acc: {mean_acc:.3f}''', end="\r")
        
    
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

