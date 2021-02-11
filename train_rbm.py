import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim

from config import *

from data import SSQAData_RBM
from pgm.layers import OneHotLayer, GaussianLayer
from pgm.model import MRF

device = torch.device('cpu')

batch_size = 300
q = 21
k = 10
lamb_l1b = 0.025
Nh = 200
DATASET = "PF00397"

train_dataset = SSQAData_RBM(f"{PFAM_DATA}/{DATASET}/data.pt", subset = "train")
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

val_dataset = SSQAData_RBM(f"{PFAM_DATA}/{DATASET}/data.pt", subset = "val")
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

_, _, N = train_dataset.seqs.size()
gamma = lamb_l1b / (2 * q * N)

pots = torch.zeros(q, N)
for v, w in train_dataset:
    pots += w*v
pots /= torch.sum(train_dataset.weights)
pots = (pots-pots.mean(0)[None]).view(-1).float().to(device)

print("Training with only sequence")

visible_layers = ["sequence"]
hidden_layers = ["hidden"]

v = OneHotLayer(pots, N=N, q=q, name="sequence")
h = GaussianLayer(N=200, name="hidden")

E = [(v.name, h.name)]

model_rbm = MRF(layers={v.name: v,
                        h.name: h}, edges=E, name="")

for visible in visible_layers:
    edge = model_rbm.get_edge(visible, "hidden")
    edge.gauge = edge.gauge.to(device)

optimizer = optim.Adam(model_rbm.parameters(), lr=0.001)

for epoch in range(40000):
    model_rbm.train_epoch(optimizer, train_loader, visible_layers, hidden_layers, [gamma], epoch,
          savepath=f"{PFAM_DATA}/{DATASET}/weights/seq-reg-200")
    if not epoch % 30:
        model_rbm.val(val_loader, visible_layers, hidden_layers, epoch)