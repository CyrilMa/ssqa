import torch
from torch.utils.data import DataLoader
from torch import optim

from data import SecondaryStructureAnnotatedDataset, collate_sequences_train
from ss_inference import NetSurfP2

from config import *

train_dataset = SecondaryStructureAnnotatedDataset(f"{UTILS}/training_set.pt", 50)
train_loader = DataLoader(train_dataset, batch_size = 15, collate_fn = collate_sequences_train,
                        shuffle = True, drop_last=True)

val_dataset = SecondaryStructureAnnotatedDataset(f"{UTILS}/validation_set.pt", 50)
val_loader = DataLoader(val_dataset, batch_size = 15, collate_fn = collate_sequences_train,
                        shuffle=False, drop_last=False)

device = torch.device('cuda')

model = NetSurfP2(50, name="netsurp2")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

max_acc = 0
for i in range(50):
    model.train_epoch(train_loader, optimizer, i)
    mean_ss3_acc, _ = model.val_epoch(val_loader, i)
    if mean_ss3_acc > max_acc:
        torch.save(model.state_dict(), f"{UTILS}/nsp2_50feats.h5")
        max_acc = mean_ss3_acc