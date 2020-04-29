import time
from structure.utils import *
from .metrics import aa_acc, l1b_reg, msa_mean

torch.cuda.is_available()
device = torch.device('cpu')
LAYERS_NAME = ["sequence", "structure", "transitions"]


def train(model, optimizer, loader, visible_layers, hidden_layers, gammas, epoch, savepath="seq100"):
    start = time.time()
    model.train()
    mean_loss, mean_reg, mean_acc = 0, 0, 0
    edges = [model.get_edge(v, "hidden") for v in visible_layers]
    for batch_idx, data in enumerate(loader):
        d_0 = {k: v.float().permute(0, 2, 1).to(device) for k, v in zip(LAYERS_NAME, data[:-2]) if k in visible_layers}
        w = data[-1].float().to(device)
        batch_size, q, N = d_0["sequence"].size()

        # Sampling
        d_0, d_f = model.gibbs_sampling(d_0, visible_layers, hidden_layers, k=10)

        # Optimization
        optimizer.zero_grad()
        e_0, e_f = model(d_0), model(d_f)
        loss = msa_mean(e_f - e_0, w).clamp(-100, 100)
        for gamma, edge in zip(gammas, edges):
            loss += gamma * l1b_reg(edge)
        loss.backward()
        optimizer.step()

        # Metrics
        d_0, d_f = model.gibbs_sampling(d_0, visible_layers, hidden_layers, k=1)
        acc = aa_acc(d_0["sequence"].view(batch_size, q, N), d_f["sequence"].view(batch_size, q, N))
        ll = msa_mean(model.integrate_likelihood(d_f, "hidden"), w).clamp(-100, 100) / 31
        mean_loss = (mean_loss * batch_idx + ll.item()) / (batch_idx + 1)
        mean_reg = (mean_reg * batch_idx) / (batch_idx + 1)
        mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)
        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60

    print(
        f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || Reg: {mean_reg:.3f} || Acc: {mean_acc:.3f}''',
        end="\r")
    if not epoch % 30:
        print(
            f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || Reg: {mean_reg:.3f} || Acc: {mean_acc:.3f}''')
        model.save(f"{savepath}_{epoch}.h5")


def val(model, loader, visible_layers, hidden_layers, epoch):
    start = time.time()
    model.eval()
    mean_loss, mean_reg, mean_acc = 0, 0, 0
    Z = model.ais()
    for batch_idx, data in enumerate(loader):
        d_0 = {k: v.float().permute(0, 2, 1).to(device) for k, v in zip(LAYERS_NAME, data[:-1]) if k in visible_layers}
        w = data[-1].float().to(device)
        batch_size, q, N = d_0["sequence"].size()
        # Sampling
        d_0, d_f = model.gibbs_sampling(d_0, visible_layers, hidden_layers, k=10)

        d_0, d_f = model.gibbs_sampling(d_0, visible_layers, hidden_layers, k=1)
        acc = aa_acc(d_0["sequence"].view(batch_size, q, N), d_f["sequence"].view(batch_size, q, N))
        ll = msa_mean(model.integrate_likelihood(d_f, "hidden"), w) / 31
        mean_loss = (mean_loss * batch_idx + ll.item()) / (batch_idx + 1)
        mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)
        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60

    print(
        f'''Val Epoch: {epoch} [100%] || Time: {m} min {s} || NLL: {mean_loss-Z:.3f} || Acc: {mean_acc:.3f}''')
