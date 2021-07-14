import torch
import torch.nn as nn
import pdb

from torch.nn.modules import dropout

class SimpleLSTM(nn.Module):
    def __init__(self, dim_input, hidden, num_layers, droprate):
        super().__init__()
        self.lstm  = nn.LSTM(dim_input, hidden, num_layers, batch_first=True, dropout=droprate)
        self.last_fc = nn.Linear(hidden, 1)


    def forward(self, xs, seqlen): # B, max_los, 29
        # if self.training

        packed = nn.utils.rnn.pack_padded_sequence(xs, lengths=seqlen, batch_first=True, enforce_sorted=False)
        packed_out, (h,c) = self.lstm(packed)

        unpacked_out, unpacked_seqlen = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        lstm_out = unpacked_out[torch.arange(xs.size(0)), unpacked_seqlen-1, :] # bsz, 32
        y_pred = self.last_fc(lstm_out)
        y_pred = torch.sigmoid(y_pred)

        return y_pred
