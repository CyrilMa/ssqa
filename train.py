import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim

from config import *

from pgm.data import SequenceStructureData
from pgm.layers import OneHotLayer, DReLULayer, GaussianLayer
from pgm.model import MRF
from pgm.train import train, val

torch.cuda.is_available()
torch.set_num_threads(8)
device = torch.device('cpu')

batch_size = 300
q = 21
N = 31
k = 10
lamb_l1b = 0.025
Nh = 200
gamma = lamb_l1b / (2 * q * N)

train_dataset = SequenceStructureData(f"{DATA}/{DATASET}", dataset="train")
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, drop_last=True)

val_dataset = SequenceStructureData(f"{DATA}/{DATASET}", dataset="val")
val_loader = DataLoader(val_dataset, batch_size=batch_size)

_, N, qx = train_dataset.raw_sequences.shape
_, _, qs = train_dataset.ss_sequences.shape
_, Nt, qt = train_dataset.ss_transitions.shape

pots = np.zeros((N, qx))
for w, v in zip(train_dataset.weights, train_dataset.raw_sequences):
    pots += w * v
pots /= np.sum(train_dataset.weights)
pots = pots.T
pots = (pots.T - np.mean(pots, 1)).T
pots = torch.tensor(pots).float().view(-1).to(device)

pots2 = np.zeros((N, qs))
for w, v in zip(train_dataset.weights, train_dataset.ss_sequences):
    pots2 += w * v
pots2 /= np.sum(train_dataset.weights)
pots2 = pots2.T
pots2 = (pots2.T - np.mean(pots2, 1)).T
pots2 = torch.tensor(pots2).float().view(-1).to(device)

pots3 = np.zeros((Nt, qt))
for w, v in zip(train_dataset.weights, train_dataset.ss_transitions):
    pots3 += w * v
pots3 /= np.sum(train_dataset.weights)
pots3 = pots3.T
pots3 = (pots3.T - np.mean(pots3, 1)).T
pots3 = torch.tensor(pots3).float().view(-1).to(device)

print("Training with only sequence")

visible_layers = ["sequence"]
hidden_layers = ["hidden"]

v = OneHotLayer(pots, N=N, q=qx, name="sequence")
h = GaussianLayer(N=Nh, name="hidden")

E = [(v.name, h.name)]

model = MRF(layers={v.name: v,
                    h.name: h},
            edges=E).to(device)

for visible in visible_layers:
    edge = model.get_edge(visible, "hidden")
    edge.gauge = edge.gauge.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 4001):
    train(model, optimizer, train_loader, visible_layers, hidden_layers, [gamma], epoch,
          savepath=f"{DATA}/{DATASET}/weights/seq-reg-200")
    if not epoch % 30:
        val(model, val_loader, visible_layers, hidden_layers, epoch)

print("Training with sequence and structure")

visible_layers = ["sequence", "structure"]
hidden_layers = ["hidden"]

v = OneHotLayer(pots, N=N, q=qx, name="sequence")
s = OneHotLayer(pots2, N=N, q=qs, name="structure")
h = GaussianLayer(N=Nh, name="hidden")

E = [(v.name, h.name),
     (s.name, h.name),
     ]

model = MRF(layers={v.name: v,
                    s.name: s,
                    h.name: h},
            edges=E).to(device)

for visible in visible_layers:
    edge = model.get_edge(visible, "hidden")
    edge.gauge = edge.gauge.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 4001):
    train(model, optimizer, train_loader, visible_layers, hidden_layers, [gamma, gamma/4], epoch,
          savepath=f"{DATA}/{DATASET}/weights/seq-struct-reg-200")
    if not epoch % 30:
        val(model, val_loader, visible_layers, hidden_layers, epoch)

print("Training with sequence, structure and length")

visible_layers = ["sequence", "structure", "transitions"]
hidden_layers = ["hidden"]

v = OneHotLayer(pots, N=N, q=qx, name="sequence")
s = OneHotLayer(pots2, N=N, q=qs, name="structure")
t = OneHotLayer(pots3, N=Nt, q=qt, name="transitions")
h = GaussianLayer(N=Nh, name="hidden")

E = [(v.name, h.name),
     (s.name, h.name),
     (t.name, h.name),
     ]

model = MRF(layers={v.name: v,
                    s.name: s,
                    t.name: t,
                    h.name: h},
            edges=E).to(device)

for visible in visible_layers:
    edge = model.get_edge(visible, "hidden")
    edge.gauge = edge.gauge.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 4001):
    train(model, optimizer, train_loader, visible_layers, hidden_layers, [gamma, gamma/4, gamma/4], epoch,
          savepath=f"{DATA}/{DATASET}/weights/seq-structure-transitions-reg-200")
    if not epoch % 30:
        val(model, val_loader, visible_layers, hidden_layers, epoch)
