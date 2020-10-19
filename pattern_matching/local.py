from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer

import torch
from torch.nn import functional as F

from ..utils import trace

def build_length()
a = torch.zeros((len(t_), len(t_[0]), len(t_[0][0]), 30))
for k, temp_t_ in enumerate(t_):
    for j in tqdm(range(30)):
        a[k, :, :, j] = trace(torch.exp(temp_t_), offset = j)