import pdb
import torch.nn as nn
import torch
from config import args
from utils.misc import set_seeds
from models.Soobin.network import LatentModel as ANP_Model
from models.Dupont.network import NeuralProcess as NP_Model
from models.Set_NP.module import PMA_shared

set_seeds(args.seed)

class SetNP_ViT(nn.Module):
    """
    Vit-based Set-NP network
    """
    def __init__(self, dim_h, dim_modal) -> None:
        super().__init__()
        self.np_model = ANP_Model(dim_h, dim_modal)

    def forward(self, context_x, context_y, target_x, target_y=None):
        y_pred, kl, loss = self.np_model(context_x, context_y, target_x, target_y)

        return y_pred, kl, loss


class SetNP(nn.Module):
    """
    Vit-based Set-NP network
    """
    def __init__(self, dim_h, dim_emr, dim_txt, dim_pos, dim_set, num_heads) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emr = dim_emr
        self.dim_txt = dim_txt
        self.dim_pos = dim_pos
        self.dim_set = dim_set


        self.fc_emr = nn.Linear(dim_emr + dim_pos, dim_set)
        self.fc_txt = nn.Linear(dim_txt + dim_pos, dim_set)
        self.set_embedder = PMA_shared(dim_set, num_heads=num_heads, num_seeds=1)
        self.np_model = ANP_Model(dim_h, dim_modal=dim_set)
        # self.np_model = NP_Model(self.dim_emr + self.dim_pos, args.dim_output, args.dim_hidden, args.dim_hidden, args.dim_hidden) # Dupont version

    def forward(self, context_x, context_y, target_x, target_y=None):
        # variable set size

        cx_emr = torch.cat((context_x[:,:, :self.dim_emr], context_x[:,:, -self.dim_pos:]), dim=-1)
        tx_emr = torch.cat((target_x[:,:, :self.dim_emr], target_x[:,:, -self.dim_pos:]), dim=-1)
        cx_txt = context_x[:,:, self.dim_emr:] # embed + pos
        tx_txt = target_x[:,:, self.dim_emr:]
        cx_emr_in = torch.unsqueeze(self.fc_emr(cx_emr), dim=2)
        tx_emr_in = torch.unsqueeze(self.fc_emr(tx_emr), dim=2)
        cx_txt_in = torch.unsqueeze(self.fc_txt(cx_txt), dim=2)
        tx_txt_in = torch.unsqueeze(self.fc_txt(tx_txt), dim=2)


        context_x_set_in = torch.cat((cx_emr_in, cx_txt_in), dim=-2) # (B, obs, setsize, 64)
        target_x_set_in = torch.cat((tx_emr_in, tx_txt_in), dim=-2)

        # observation-wise set embedding
        cx_set = self.set_embedder(context_x_set_in) # output shape: (B, num_seeds, dim_h)
        tx_set = self.set_embedder(target_x_set_in) # output shape: (B, num_seeds, dim_h)

        y_pred, kl, loss = self.np_model(cx_set, context_y, tx_set, target_y)

        return y_pred, kl, loss
