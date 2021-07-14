import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def expanded_pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix (torch.tensor)
           y is an optional Mxd matirx (torch.tensor)
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
         differences = x.unsqueeze(1) - y.unsqueeze(0) # N,M,d
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0) # N,N,d
    distances = torch.sum(differences * differences, -1) # N, M(N)
    return distances

def LDA_Loss(embedding, label, cv_type=1):
    '''
    emedding: , label: MxN
    '''
    pos_idx = torch.eq(label,1.0) # (num_positive->max to M*N,2)
    neg_idx = torch.eq(label,0.0) # (num_negative,2)

    pos_idx = pos_idx[:,0]
    neg_idx = neg_idx[:,0]

    pos_embedding = embedding[pos_idx] # (num_positive, embedding)
    neg_embedding = embedding[neg_idx]

    pos_distance = torch.sum(expanded_pairwise_distances(pos_embedding))
    neg_distance = torch.sum(expanded_pairwise_distances(neg_embedding))

    return pos_distance + neg_distance


class BiLSTM(nn.Module):
    def __init__(self, dim_input, hidden, num_layers, droprate):
        super().__init__()
        self.lstm  = nn.LSTM(dim_input, hidden, num_layers, batch_first=True, dropout=droprate, bidirectional=True)
        self.last_fc = nn.Linear(hidden*2, 1)

    def forward(self, xs, seqlen): # B, max_los, 29
        packed = nn.utils.rnn.pack_padded_sequence(xs, lengths=seqlen, batch_first=True, enforce_sorted=False)
        packed_out, (h,c) = self.lstm(packed)

        unpacked_out, unpacked_seqlen = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) # (16,72,64), (16)
        lstm_out = unpacked_out[torch.arange(xs.size(0)), unpacked_seqlen-1, :] # bsz, 64
        y_pred = self.last_fc(lstm_out)
        y_pred = torch.sigmoid(y_pred)

        return y_pred

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        batch_size = inputs.shape[0]
        weights = torch.bmm(inputs.unsqueeze(1),
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
        # weights = torch.bmm(inputs.unsqueeze(-1),
        #                     self.att_weights  # (1, hidden_size)
        #                     .permute(1, 0)  # (hidden_size, 1)
        #                     .unsqueeze(0)  # (1, hidden_size, 1)
        #                     .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
        #                     .permute(0, 2, 1)
        #                     )
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        _sums = attentions.sum(-1).unsqueeze(-1)
        attentions = attentions.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions

class BiLSTM_Att(nn.Module):
    def __init__(self, dim_input, hidden, num_layers, droprate):
        super().__init__()
        self.lstm  = nn.LSTM(dim_input, hidden, num_layers, batch_first=True, dropout=droprate, bidirectional=True)
        self.last_fc = nn.Linear(hidden*2, 1)

        self.attention = Attention(hidden*2)

    def forward(self, xs, seqlen): # B, max_los, 29
        packed = nn.utils.rnn.pack_padded_sequence(xs, lengths=seqlen, batch_first=True, enforce_sorted=False)
        packed_out, (h,c) = self.lstm(packed)

        unpacked_out, unpacked_seqlen = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) # (16,72,64), (16)
        lstm_out = unpacked_out[torch.arange(xs.size(0)), unpacked_seqlen-1, :] # bsz, 64
        lstm_out, attentions = self.attention(lstm_out, unpacked_seqlen)
        # pdb.set_trace()
        # y_pred = self.last_fc(lstm_out)
        y_pred = torch.sigmoid(lstm_out)
        return y_pred.unsqueeze(-1)


class DynamicBiLSTM(nn.Module): # from tf code
    def __init__(self, dim_input, hidden, num_layers, droprate):
        super().__init__()
        self.lstm  = nn.LSTM(dim_input, hidden, num_layers, batch_first=True, dropout=droprate)

        # self.MAX_SEQ_LENGTH = 71
        # self.N_UNIQUE_DYNAMIC_FEATURES = 29

        self.n_labels = 1
        self.lambda_reg = 0.01
        # self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.lambda_reg)
        self.net_static0 = nn.Sequential(
                           nn.ZeroPad2d((1, 2, 1, 2)),
                           nn.Conv2d(in_channels=dim_input, out_channels=hidden, kernel_size=(1,1)),
                           nn.LeakyReLU(),
                        )
        self.net_static1 = nn.Sequential(
                           nn.ZeroPad2d((1, 2, 1, 2)),
                           nn.Conv2d(in_channels=hidden, out_channels=64, kernel_size=(1,1)),
                           nn.LeakyReLU(),
                        )
        self.stack_bidirectional_dynamic_rnn = nn.LSTM(dim_input, hidden, num_layers, batch_first=True, dropout=droprate)

        self.last_fc = nn.Linear(hidden*2, 1)

    def forward(self, xs, seqlen): # B, max_los, 29
        # if self.training
        packed = nn.utils.rnn.pack_padded_sequence(xs, lengths=seqlen, batch_first=True, enforce_sorted=False)
        packed_out, (h,c) = self.lstm(packed)

        # (bsz,max_los,32)
        unpacked_out, unpacked_seqlen = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        lstm_out = unpacked_out[torch.arange(xs.size(0)), unpacked_seqlen-1, :] # bsz, 32
        # att:
        y_pred = self.last_fc(lstm_out)
        y_pred = torch.sigmoid(y_pred)

        ###
        attention_score = self.stack_bidirectional_dynamic_rnn()

        return y_pred
