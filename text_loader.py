import logging
import pdb
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from config import args

class Embedder:
    def __init__(self, model, device, output_path: str=""):
        self.model = model.to(device)
        self.output_path = output_path
        self.device = device

    def formalize_sequence(self, text: list, max_seq_len):
        '''
        text: list of token indices
        output: {'input_ids': torch.Size([1, 512]),
                 'token_type_ids': torch.Size([1, 512]),
                 'attention_mask': torch.Size([1, 512])}
        '''
        # pad sequence
        original_len = len(text) if len(text) < max_seq_len else max_seq_len
        pad_len = max_seq_len - original_len
        text = text[:original_len] + [0] * pad_len
        segment_ids = [1] * max_seq_len
        input_mask = [1] * original_len + [0] * pad_len

        output = dict()
        output.update({'input_ids': torch.LongTensor(text).reshape(1, -1).to(self.device)})
        output.update({'token_type_ids': torch.LongTensor(segment_ids).reshape(1, -1).to(self.device)})
        output.update({'attention_mask': torch.LongTensor(input_mask).reshape(1, -1).to(self.device)})
        return output

    def _embedding_loop(self, text):
        input_seq = self.formalize_sequence(text, args.max_seq_len)

        model = self.model
        model.eval()
        with torch.no_grad():
            outputs = model(**input_seq) # fix
        last_hidden_states = outputs[0].detach()

        embeddings = []
        if args.pooling == 'first':
            embeddings = last_hidden_states[:,0,:]
        elif args.pooling == 'sum' or args.pooling == 'mean':
            # masking [CLS] and [SEP]
            attention_mask = input_seq['attention_mask'].detach()
            attention_mask = nn.functional.pad(attention_mask[:,2:],(1,1)) # 2 means [CLS] and [SEP]

            # extract the hidden state where there's no masking
            attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.shape)
            sub_embeddings = (attention_mask.to(torch.float)*last_hidden_states)

            # summation
            embeddings = sub_embeddings.sum(dim=1)

            # mean
            if args.pooling == 'mean':
                attention_mask = attention_mask[:,:,0].sum(dim=-1).unsqueeze(1)
                embeddings = embeddings/attention_mask.to(torch.float)

        elif args.pooling == 'none':
            for embed, attention_mask in zip(last_hidden_states, inputs['attention_mask']):
                token_embed = embed[0:attention_mask.sum()]
                embeddings.append(token_embed)

        if args.pooling != 'none':
            # if 'none', return a set of token vectors.
            return embeddings.cpu().reshape(-1)
# eri
