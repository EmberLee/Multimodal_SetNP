import os; import sys
import numpy as np; import pandas as pd
import pickle as pkl
import argparse
import pdb

from tqdm import tqdm
from config import args
import torch as t
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel,
    HfArgumentParser, set_seed,
)
from text_loader import Embedder

def main(split='val', result=None):
    data = pkl.load(open(f'data/MIMIC/mmnp_xypair_split/multimodal_MIMIC_{split}_xypair.pkl', 'rb'))
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    embedder = Embedder(model, device)
    if not result:
        result = dict()
    pdb.set_trace()
    with tqdm(enumerate(data), total=len(data), desc='Making Embeddings...') as tq:
        for i, infos in tq:
            id = infos['CHID']
            valid = infos['TXT_VALID_STEPS']
            tokens = infos['TXT']
            if len(valid) == 0:
                continue
            if result.get(id):
                print('ID already exists.')
                pdb.set_trace()

            result.update({id: t.zeros([len(valid), 768])})
            for j, step in enumerate(valid):
                result[id][j] = embedder._embedding_loop(tokens[step])

    if split == 'test':
        pkl.dump(result, open('data/embeddings.pkl', 'wb'))
    # pdb.set_trace()
    print('Finished')
    return result

def main2():
    data = pkl.load(open(f'data/MIMIC/mmnp_xypair/multimodal_MIMIC_mmnp_xypair.pkl', 'rb'))
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    embedder = Embedder(model, device)

    result = dict()
    with tqdm(enumerate(data), total=len(data), desc='Making Embeddings...') as tq:
        for i, infos in tq:
            id = infos['CHID']
            valid = infos['TXT_VALID_STEPS']
            tokens = infos['TXT']
            if len(valid) == 0:
                continue
            if result.get(id):
                print('ID already exists.')
                pdb.set_trace()

            result.update({id: t.zeros([len(valid), 768])})
            for j, step in enumerate(valid):
                result[id][j] = embedder._embedding_loop(tokens[step])

    pkl.dump(result, open('data/embeddings.pkl', 'wb'))
    # pdb.set_trace()
    print('Finished')
    return result

if __name__=='__main__':
    # splits = ['val', 'train', 'test']
    # result = None
    # for splt in splits:
    #     result = main(splt, result)
    main2()
