import torch
import pickle as pkl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import ModuleList, ModuleDict, Sequential

class FC(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super(FC, self).__init__()
        ## use nn.Module or ModuleDict() or nn.ModuleList() to wrap those submodules.
        self.encoder = nn.Linear(d_in, d_h) # should be a nn.Transformer
        self.cross_att = nn.Linear(d_h, d_out)
    
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        x = self.cross_att(x)
        x = torch.sigmoid(x)
        return x


# class NCF(nn.Module):
#     pass

# class Extractor_IMG:
# class Extractor_TXT:
#     pass

