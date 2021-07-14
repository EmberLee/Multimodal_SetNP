import os; import sys
import numpy as np; import pandas as pd
import pdb
import pickle as pkl

from tqdm import tqdm
import torch as t

data = pkl.load(open('data/MIMIC/multimodal_MIMIC_mmnp_xypair.pkl', 'rb'))

ids = list(map(lambda x: x['CHID'], data))

EMPTY_IDS = ['173966', '114279', '170790', '105663', '100311']
EMPTY_STEPS = [46, 16, 135, 247, 47]

for i, id in enumerate(EMPTY_IDS):
    valid_steps = data[ids.index(id)]['TXT_VALID_STEPS']
    # valid_steps.remove(int(EMPTY_STEPS[i]))
    # data[ids.index(id)]['TXT_VALID_STEPS'] = valid_steps
    print(valid_steps)
# pkl.dump(data, open('data/MIMIC/multimodal_MIMIC_mmnp_xypair_bert.pkl', 'wb'))
print('Finished')
