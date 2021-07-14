import time

import torch as t
import pickle as pkl
import numpy as np
import pdb
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset, sampler
from config import args
from utils.collate import collate_fn

def exclude_id(data: np.array, modality):
    '''
    exclude ids whose valid time step length < 2
    before applying K fold
    '''
    idxes = list(range(len(data)))
    if modality == 'emr':
        return data
    elif modality == 'txt':
        for i, v in enumerate(data):
            if len(v['TXT_VALID_STEPS']) < 2:
                idxes.remove(i)
        return data[np.array(idxes)]
    elif modality == 'both':
        for i, v in enumerate(data):
            valid_steps = list(set(v['EMR_VALID_STEPS'] + v['TXT_VALID_STEPS']))
            valid_steps.sort()
            if len(v['TXT_VALID_STEPS']) < 2:
                idxes.remove(i)
        return data[np.array(idxes)]

def get_Modal_loader(args_input, dataset, balancing=False):
    if args.is_imputed:
        reader = ModalityReader_Imputed(dataset=dataset, modality=args_input.modality)
        dim_modal = reader._x_size
    else:
        reader = ModalityReader(dataset=dataset, modality=args_input.modality)
        dim_modal = reader._x_size

    # balancing
    if balancing:
        targets = np.array(reader.has_onset, dtype=np.int32)
        class_sample_cnt = np.unique(targets, return_counts=True)[1] # # of [0s, 1s]
        weights = 1. / class_sample_cnt
        samples_weight = weights[targets]
        samples_weight = t.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
        if args.is_imputed:
            loader = DataLoader(reader,
                                batch_size=args.batch_size, drop_last=False, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True)
        else:
            loader = DataLoader(reader,
                                batch_size=args.batch_size, drop_last=False, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        if args.is_imputed:
            loader = DataLoader(reader,
                                batch_size=args.batch_size, drop_last=False, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)
        else:
            loader = DataLoader(reader,
                                batch_size=args.batch_size, drop_last=False, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    return loader, dim_modal


## convert to torch dataset
class ModalityReader(Dataset):
    '''
    Make Reader like GPCurveReader class
    '''

    def __init__(self,
                dataset: np.array,
                testing=False,
                modality='emr',
                device=t.device("cuda:0" if t.cuda.is_available() else "cpu"),
                embedding_path=None
                ):

        modality = modality.upper()
        assert modality in ['EMR', 'TXT', 'BOTH'] # choose certain modality to load.

        if modality == 'EMR':
            self._x_size = args.dim_emr_feats + args.dim_pos
        elif modality == 'TXT':
            self._x_size = args.dim_embed + args.dim_pos
        elif modality == 'BOTH':
            self._x_size = args.dim_emr_feats + args.dim_embed + args.dim_pos

        self._y_size = args.dim_output
        self._event_idx = args.task_idx

        # dict_keys(['SEQ_LENGTH', 'POS', 'ICU_IN', 'ICU_OUT', 'AGE', 'GENDER', 'LABEL', 'TXT', 'EMR', 'TXT_VALID_STEPS', 'EMR_VALID_STEPS', 'CHID'])
        self._data_allkeys = dataset
        self._data_length = len(self._data_allkeys) # if no data observed (e.g. TXT), that instance is excluded.

        self.features = []
        self.labels_emr = []
        self.pos_mat = []
        self.has_onset = []

        # {chid: t.Tensor([len_txt_valid_steps, 768]), ...}
        if args.use_embedding_file:
            with open(args.embedding_path, 'rb') as f:
                self.embeddings = pkl.load(f)

        with tqdm(enumerate(self._data_allkeys), total=len(self._data_allkeys), desc='Initializing dataloader => Organizing time-series') as tq:
            for idx, v in tq:
                if modality == 'EMR':
                    valid_steps = v['EMR_VALID_STEPS'] # list of time steps: [0,1,2,6, ...]
                    valid_infos = v['EMR']

                elif modality == 'TXT':
                    valid_steps = v['TXT_VALID_STEPS'] # list of time steps: [0,1,2,6, ...]
                    if len(valid_steps) < 2: # valid time step이 2 이상인 환자만, 아닐 경우 context와 target이 겹치기 때문
                        self._data_length -= 1
                        continue

                elif modality == 'BOTH':
                    valid_steps = list(set(v['EMR_VALID_STEPS'] + v['TXT_VALID_STEPS']))
                    valid_steps.sort()

                tmparr = t.zeros(len(valid_steps), self._x_size)
                for i, step in enumerate(valid_steps):
                    if modality == 'EMR':
                        e = t.Tensor(valid_infos[step].tolist()).type(t.float32)
                    elif modality == 'TXT':
                        if args.use_embedding_file:
                            e = self.embeddings[v['CHID']][valid_steps.index(step)] # 임베딩 파일 새로 만들어야함
                        else:
                            e = v['EMBEDDING'][step]
                    elif modality == 'BOTH':
                        e = t.zeros([args.dim_emr_feats + args.dim_embed]).type(t.float32)
                        if step in v['EMR_VALID_STEPS']:
                            e[:args.dim_emr_feats] = t.Tensor(v['EMR'][step].tolist()).type(t.float32)
                        if step in v['TXT_VALID_STEPS']:
                            if args.use_embedding_file:
                                e[args.dim_emr_feats:] = self.embeddings[v['CHID']][v['TXT_VALID_STEPS'].index(step)]
                            else:
                                e[args.dim_emr_feats:] = v['EMBEDDING'][step]
                    p = t.Tensor(v['POS'][step]).type(t.float32)
                    tmparr[i] = t.cat((e, p))

                self.features.append(tmparr) # feats_emr, feats_txt, feats_both
                self.labels_emr.append(t.Tensor(v['LABEL'][valid_steps]).type(t.float32))
                self.pos_mat.append(t.Tensor(v['POS'][valid_steps]).type(t.float32))
                self.has_onset.append(self.labels_emr[-1][-1, self._event_idx])
            # pdb.set_trace()
            self.features = np.array(self.features, dtype='object')
            self.labels_emr = np.array(self.labels_emr, dtype='object')
            self.pos_mat = np.array(self.pos_mat, dtype='object')
            self.has_onset = np.array(self.has_onset)


    def __len__(self):
        return self._data_length

    def __getitem__(self, index):
        return self.features[index][:, :self._x_size], \
               self.labels_emr[index][:, self._event_idx], \
               self.pos_mat[index]

## for baseline
class ModalityReader_Imputed(Dataset):
    def __init__(self,
                 dataset: np.array,
                 modality='emr',
                 ):

        modality = modality.upper()
        assert modality in ['EMR', 'TXT', 'BOTH'] # choose certain modality to load.

        if modality == 'EMR':
            self._x_size = args.dim_emr_feats
        elif modality == 'TXT':
            self._x_size = args.dim_embed
        elif modality == 'BOTH':
            self._x_size = args.dim_emr_feats + args.dim_embed

        self._y_size = args.dim_output
        self._event_idx = args.task_idx

        # dict_keys(['SEQ_LENGTH', 'POS', 'ICU_IN', 'ICU_OUT', 'AGE', 'GENDER', 'LABEL', 'TXT', 'EMR', 'TXT_VALID_STEPS', 'EMR_VALID_STEPS', 'CHID'])
        self._data_allkeys = dataset
        self._data_length = len(self._data_allkeys) # if no data observed (e.g. TXT), that instance is excluded.

        self.features = []
        self.labels = []
        self.has_onset = []
        self.onset_indices = []

        # {chid: t.Tensor([len_txt_valid_steps, 768]), ...}
        if args.use_embedding_file:
            with open(args.embedding_path, 'rb') as f:
                self.embeddings = pkl.load(f)

        with tqdm(enumerate(self._data_allkeys), total=len(self._data_allkeys), desc='Initializing dataloader => Organizing time-series-imputed') as tq:
            for idx, v in tq:
                if modality == 'EMR':
                    valid_steps = v['EMR_VALID_STEPS'] # list of time steps: [0,1,2,6, ...]
                    valid_infos = v['EMR']
                elif modality == 'TXT':
                    valid_steps = v['TXT_VALID_STEPS'] # list of time steps: [0,1,2,6, ...]
                    valid_infos = v['TXT']
                    if len(valid_steps) < 2: # valid time step이 2 이상인 환자만, 아닐 경우 context와 target이 겹치기 때문
                        self._data_length -= 1
                        continue
                elif modality == 'BOTH':
                    valid_steps = list(set(v['EMR_VALID_STEPS'] + v['TXT_VALID_STEPS']))
                    valid_steps.sort()
                    valid_infos = v['EMR']
                    valid_infos_txt = v['TXT']

                tmparr = t.zeros(len(valid_infos), self._x_size)

                embed_idx = -1
                for i, emr_step in enumerate(valid_infos):
                    if modality == 'EMR':
                        e = t.Tensor(emr_step.tolist()).type(t.float32)
                    elif modality == 'TXT':
                        if args.use_embedding_file:
                            next_validstep = v['TXT_VALID_STEPS'][min(embed_idx+1, len(v['TXT_VALID_STEPS']) -1 )]
                            if i == next_validstep:
                                embed_idx += 1
                            if embed_idx == -1: # before the first note has observed
                                continue
                            else:
                                e = self.embeddings[v['CHID']][embed_idx] # validsteps = [2, 6, 12, ...] # index shifting
                        else:
                            e = v['EMBEDDING'][i]

                    elif modality == 'BOTH':
                        e = t.zeros([self._x_size]).type(t.float32)
                        # EMR
                        if len(v['EMR_VALID_STEPS']) > 0:
                            e[:args.dim_emr_feats] = t.Tensor(emr_step.tolist()).type(t.float32)
                        # TXT
                        if len(v['TXT_VALID_STEPS']) > 0:
                            if args.use_embedding_file:
                                next_validstep = v['TXT_VALID_STEPS'][min(embed_idx+1, len(v['TXT_VALID_STEPS']) -1 )]
                                if i == next_validstep:
                                    embed_idx += 1
                                if embed_idx == -1: # before the first note has observed
                                    continue
                                else:
                                    e[args.dim_emr_feats:] = self.embeddings[v['CHID']][embed_idx] # validsteps = [2, 6, 12, ...] # index shifting
                            else:
                                e[args.dim_emr_feats:] = v['EMBEDDING'][i]

                    tmparr[i] = e


                self.features.append(tmparr)
                self.labels.append(t.Tensor(v['LABEL'][:, self._event_idx]).type(t.float32))
                self.onset_indices.append(t.argmax(t.Tensor(v['LABEL'][:, self._event_idx])))
                self.has_onset.append(self.labels[-1][-1])

            self.features = np.array(self.features, dtype='object') # variable length
            self.labels = np.array(self.labels, dtype='object') # variable length
            self.has_onset = np.array(self.has_onset)


    def __len__(self):
        return self._data_length

    def __getitem__(self, index):
        maxlen = args.max_los

        out_feat = t.zeros(maxlen, self._x_size)
        out_label = t.zeros(1)
        out_lens = t.zeros(1)

        this_feat = self.features[index]
        this_label = self.labels[index]

        pred_bf = t.randint(1, args.pred_before+1, size=(1,))[0] # WITHIN task
        len_lab_m1 = t.Tensor([len(this_feat)-1]).type(t.int32)[0]

        onset_time = self.onset_indices[index]
        if onset_time < 1:
            onset_time = t.randint(int(t.minimum(pred_bf, len_lab_m1)), len_lab_m1 + 1, size=(1,))[0]

        # maxlen truncating
        x_base_idx = t.max(onset_time - pred_bf, t.Tensor([1])).type(t.int32)[0] # 2 - 6 = -4, -> 1
        if x_base_idx > maxlen:
            out_feat[:] = this_feat[x_base_idx - maxlen:x_base_idx]
            out_lens = t.Tensor([maxlen]).type(t.int32)[0]
        else:
            out_feat[:x_base_idx] = this_feat[:x_base_idx]
            out_lens = x_base_idx
        out_label = t.Tensor([this_label[-1]])

        return out_feat, out_label, out_lens


if __name__ == "__main__":
    args.dataset_path = './data/MIMIC/mmnp_xypair/multimodal_MIMIC_mmnp_xypair.pkl'
    # args.modality = 'both'
    print(args)
    with open(args.dataset_path, 'rb') as data_file:
        dset = pkl.load(data_file) # bsz, keys

    loader, dim_modal = get_Modal_loader(args, dset, balancing=False)

    with tqdm(loader, desc='Pick datum out of loader') as tq:
        tic = time.time()
        for i, data in enumerate(tq):
            if args.is_imputed:
                xs, ys, seqlen = data
                if i == 2: # for checking
                    print(xs.shape, ys.shape, seqlen.shape)
                    pdb.set_trace()
            else:
                cx, cy, tx, ty, pos_c, pos_t = data
                '''
                (16,len_context,61), (16,len_context,1), (16,1,61), (16,1,1),
                (16, len_context, 32), (16, 1, 32)
                '''
                if i == 2: # for checking
                    print(cx.shape, cy.shape, tx.shape, ty.shape, pos_c.shape, pos_t.shape)
                    pdb.set_trace()

            if i % 10 == 0:
                toc = time.time()
                print(f'{i}/{len(loader)}')

    print("EOF")
    print(f"time diff : {toc - tic:.4f} sec")
