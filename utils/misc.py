import pdb
import json
import numpy as np
import torch
import random
import shutil
import os
import logging
import logging.handlers
import pickle as pkl
import sys
from types import ModuleType, FunctionType
from gc import get_referents

def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_devices(args):
    if args.cpu or not torch.cuda.is_available():
        print("Set device to cpu")
        return torch.device('cpu')
    else:
        print("Set device to cuda")
        return torch.device('cuda')

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    ref: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)



def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Example:
        save_checkpoint({'epoch': epoch + 1,
                                'score': auroc,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint, exist_ok=True)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, is_best=True):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) exp directory which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if is_best:
        print('Load Best model')
        restore_path = os.path.join(checkpoint, 'best.pth.tar')
    else:
        print('Load Last model')
        restore_path = os.path.join(checkpoint, 'last.pth.tar')

    if not os.path.exists(restore_path):
        raise("File doesn't exist {}".format(restore_path))

    ckpt = torch.load(restore_path)
    model.load_state_dict(ckpt['state_dict'])

    start_epoch = ckpt['epoch']
    best_score = ckpt['score']

    if optimizer:
        optimizer.load_state_dict(ckpt['optim_dict'])

    return ckpt, best_score, start_epoch



"""
Define get_logger() function.
"""
def get_logger(name=None, level="DEBUG", print_level="DEBUG", file_level=None):
    """Return a customized convinience logger.

    Example:
        logger = get_logger(print_level="DEBUG", slack_level="INFO",
                            file_level="DEBUG")
        try:
            work()
        except Exception:
            # logger.exception will catch the Exception automatically within
            # the except block.
            logger.exception("An exception has occurred.")

    Args:
        name: The logger name.
        logger(str): The logging level for the logger.
        print_level(str or None): The logging level for the `StreamHandler`.
            `None` will not attach the handler.
        file_level(str or None): The logging level for the daily
            `TimedRotatingFileHandler`. `None` will not attach the handler.

    Returns:
        A logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter_full = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(message)s (%(filename)s:%(lineno)s)')

    formatter_brief = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(message)s')

    if print_level.upper() == 'INFO':
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(print_level)
        stream_handler.setFormatter(formatter_brief)
        logger.addHandler(stream_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(print_level)
        stream_handler.setFormatter(formatter_full)
        logger.addHandler(stream_handler)


    if not os.path.exists('./logs'):
        os.makedirs('./logs', exist_ok=True)
    if file_level.upper() == 'INFO':
        file_handler = logging.handlers.TimedRotatingFileHandler(
            f'./logs/mmnp_{str(name)}.log', when='D', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter_brief)
        logger.addHandler(file_handler)
    elif file_level.upper() == 'DEBUG':
        file_handler = logging.handlers.TimedRotatingFileHandler(
            f'./logs/mmnp_{str(name)}.log', when='D', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter_full)
        logger.addHandler(file_handler)

    return logger




def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def parse_label_w_time_single_event(lbl, ev_idx=0):
    res = []
    if np.sum(lbl[:, ev_idx]) > 0:
        res.append(str(1))
    else:
        res.append(str(0))
    return ''.join(res)


def label_case_split_single(dic, ev_idx):
    data = []
    data_raw = []
    labels = []
    chids = []
    cases = {} # []

    for idx, chid in enumerate(dic):
        dic[chid]['CHID'] = chid
        data.append(dic[chid])
        # data_raw.append(dic_raw[chid])
        thislabel = parse_label_w_time_single_event(dic[chid]['LABEL'], ev_idx=ev_idx)
        # cases.append(thislabel)
        if thislabel not in cases:
            cases[thislabel] = []
            cases[thislabel].append(idx)
        else:
            cases[thislabel].append(idx)

        labels.append(int(thislabel))
        chids.append(chid)

    data = np.asarray(data, dtype=object)
    # data_raw = np.asarray(data_raw, dtype=object)
    labels = np.array(labels)
    chids = np.asarray(chids)

    # return data, data_raw, labels, cases, chids
    return data, labels, cases, chids



# def reset_params(model):
#     for layer in model.children():
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()
#     return model

if __name__ == "__main__":
    print('main')
