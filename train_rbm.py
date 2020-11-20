import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim

from config import *

from pgm.data import SequenceData
from pgm.layers import OneHotLayer, GaussianLayer
from pgm.model import MRF

device = torch.device('cpu')

batch_size = 300
q = 21
k = 10
lamb_l1b = 0.025
Nh = 200
DATASET = "PF00017"

train_dataset = SequenceData(f"{DATA}/{DATASET}", dataset="train")
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, drop_last=True)

val_dataset = SequenceData(f"{DATA}/{DATASET}", dataset="val")
val_loader = DataLoader(val_dataset, batch_size=batch_size)

_, N, qx = train_dataset.raw_sequences.shape
gamma = lamb_l1b / (2 * q * N)

pots = np.zeros((N, qx))
for w, v in zip(train_dataset.weights, train_dataset.raw_sequences):
    pots += w * v
pots /= np.sum(train_dataset.weights)
pots = pots.T
pots = (pots.T - np.mean(pots, 1)).T
pots = torch.tensor(pots).float().reshape(-1).to(device)

print("Training with only sequence")

visible_layers = ["sequence"]
hidden_layers = ["hidden"]

v = OneHotLayer(pots, N=N, q=qx, name="sequence")
h = GaussianLayer(N=Nh, name="hidden")

E = [(v.name, h.name)]

model = MRF(layers={v.name: v,
                    h.name: h},
            edges=E, name = DATASET)

for visible in visible_layers:
    edge = model.get_edge(visible, "hidden")
    edge.gauge = edge.gauge.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(4000):
    model.train_epoch(optimizer, train_loader, visible_layers, hidden_layers, [gamma], epoch,
          savepath=f"{DATA}/{DATASET}/weights/seq-reg-200")
    if not epoch % 30:
        model.val(val_loader, visible_layers, hidden_layers, epoch)