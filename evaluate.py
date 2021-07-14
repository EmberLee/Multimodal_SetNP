import numpy as np
import os
import pickle as pkl

# import logging
from config import args
from utils.misc import *
from utils.metrics import auroc, auprc, get_best_f1
from data_loader import get_Modal_loader
from models.Soobin.network import LatentModel as ANP_Model
from models.Dupont.network import NeuralProcess
from models.Set_NP.network import SetNP_ViT, SetNP
from baselines.LSTM.network import SimpleLSTM
from baselines.torch_BiLSTM.network import BiLSTM
from baselines.torch_BiLSTM.network import BiLSTM_Att

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold, train_test_split


set_seeds(args.seed)


def evaluate(dataloader, model, epoch, patience, loss_fn, device, logger, task_idx, phase='validation', fold_idx=0):

    model.eval()

    preds = []
    ys = []
    losses = []

    for i, data in enumerate(dataloader):
        if args.is_imputed:
            xs, labels, seqlens = data
            if device.type == 'cuda':
                xs = xs.to(device)
                labels = labels.to(device)

            if args.model_type.upper() in ['LSTM','BILSTM','BILSTM_ATT']:
                y_pred = model(xs, seqlens)

            y = labels

        else:
            context_x, context_y, target_x, target_y, pos_c, pos_t = data
            if device.type == 'cuda':
                context_x = context_x.to(device)
                context_y = context_y.to(device)
                target_x = target_x.to(device)
                target_y = target_y.to(device)

            # pass through the latent model
            if args.model_type.upper() == 'SOOBIN':
                y_pred, kl, loss = model(context_x, context_y, target_x)
            elif args.model_type.upper() == 'DUPONT':
                y_pred_dist, y_pred, y_pred_sigma, loss = model(context_x, context_y, target_x)
            elif args.model_type.upper() == 'SETNP_VIT':
                y_pred, kl, loss = model(context_x, context_y, target_x)
            elif args.model_type.upper() == 'SETNP':
                # y_pred, kl, loss = model(context_x, context_y, target_x)
                y_pred, kl, loss = model(pos_c, pos_t, context_x, context_y, target_x, target_y)

            y_pred = torch.squeeze(y_pred[:, -1])
            y = torch.squeeze(target_y[:,-1])

        loss = loss_fn(y_pred, y)

        ys.extend(y.detach().cpu().numpy())
        preds.extend(y_pred.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())

    res_auroc = auroc(ys, preds)
    res_auprc = auprc(ys, preds)

    res_f1, residuals = get_best_f1(ys, preds) # best pr, rc, f1, all

    if logger:
        logger.info(f"[fold{fold_idx} {phase}] name:{args.exp_name}, pred_offset:{args.pred_before}, "
                    f"patience:{patience}/{args.max_patience}, "
                    f"task:{task_idx}, "
                    f"epoch: {epoch}, loss: {np.mean(losses):.4f}, "
                    f"AUROC: {res_auroc:.4f}, "
                    f"AUPRC: {res_auprc:.4f}, "
                    f"F1: {res_f1:.4f} at th {residuals[-1,0]:.4f}, "
                    )
    else:
        print(f"[fold{fold_idx} {phase}] name:{args.exp_name}, pred_offset:{args.pred_before}, "
                    f"patience:{patience}/{args.max_patience}, "
                    f"task:{task_idx}, "
                    f"epoch: {epoch}, loss: {np.mean(losses):.4f}, "
                    f"AUROC: {res_auroc:.4f}, "
                    f"AUPRC: {res_auprc:.4f}, "
                    f"F1: {res_f1:.4f} at th {residuals[-1,0]:.4f}, "
                    )


    results = {'auroc': res_auroc, 'auprc': res_auprc, 'best_f1': res_f1, 'loss': np.mean(losses)}
    return results, logger


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.pilot_activate:
        args.max_epoch = 2
        args.max_patience = 2
        args.dataset_path = './data/MIMIC/mmnp_xypair/multimodal_MIMIC_mmnp_xypair_imputed_sample.pkl'

    print("\nEvaluate the model..")
    print(args)

    if args.log_level.upper() == 'INFO':
        logger = get_logger(name=args.exp_name, print_level='INFO', file_level='INFO')
    else:
        logger = get_logger(name=args.exp_name, print_level='DEBUG', file_level='DEBUG')

    set_seeds(args.seed)

    device = set_devices(args)


    # Load whole data
    with open(args.dataset_path, 'rb') as data_file:
        data = pkl.load(data_file) # bsz, keys

    # train test split
    lbs = np.array([int(v['LABEL'][-1,0]) for v in data])
    data_tr, data_test, labels_tr, labels_test = train_test_split(data, lbs, test_size=0.1, random_state=args.seed, stratify=lbs)

    # K Fold split on train data
    skf = StratifiedKFold(n_splits=args.num_kfold, shuffle=True, random_state=args.seed)

    res_dict_val = {'auroc': [], 'auprc': [], 'best_f1': [], 'loss': []}
    res_dict_test = {'auroc': [], 'auprc': [], 'best_f1': [], 'loss': []}

    for fidx, (train_idx, val_idx) in enumerate(skf.split(data_tr, labels_tr)):
        val_loader, dim_modal = get_Modal_loader(args, data_tr[val_idx])
        test_loader, _ = get_Modal_loader(args, data_test)


        if args.model_type.upper() == 'SOOBIN':
            model = ANP_Model(args.dim_hidden, dim_modal)
        elif args.model_type.upper() == 'DUPONT':
            model = NeuralProcess(args.dim_emr_feats + args.dim_pos, args.dim_output, args.dim_hidden, args.dim_hidden, args.dim_hidden)
        elif args.model_type.upper() == 'SETNP_VIT':
            model = SetNP_ViT(args.dim_hidden)
        elif args.model_type.upper() == 'SETNP':
            model = SetNP(args.dim_hidden, args.dim_emr_feats, args.dim_embed, args.dim_pos, args.dim_set, args.num_heads_pma)
        elif args.model_type.upper() == 'LSTM':
            model = SimpleLSTM(dim_modal, args.dim_hidden, args.lstm_layers, args.drop_rate)
        elif args.model_type.upper() == 'BILSTM':
            model = BiLSTM(dim_modal, args.dim_hidden, args.lstm_layers, args.drop_rate)
        elif args.model_type.upper() == 'BILSTM_ATT':
            model = BiLSTM_Att(dim_modal, args.dim_hidden, args.lstm_layers, args.drop_rate)
        else:
            raise NotImplementedError()


        ckpt, best_score, start_epoch = load_checkpoint(os.path.join(args.save_path, args.exp_name, f'fold{fidx}'), model, is_best=not args.is_last)

        if torch.cuda.is_available():
            model.to(device)
            cudnn.benchmark = False

        loss_fn = nn.BCELoss()

        results, logger = evaluate(val_loader, model, start_epoch, 0, loss_fn, device, logger, args.task_idx, phase='val', fold_idx=fidx)
        res_dict_val['auroc'].append(results['auroc'])
        res_dict_val['auprc'].append(results['auprc'])
        res_dict_val['best_f1'].append(results['best_f1'])
        res_dict_val['loss'].append(results['loss'])

        test_results, logger = evaluate(test_loader, model, start_epoch, 0, loss_fn, device, logger, args.task_idx, phase='test', fold_idx=fidx)
        res_dict_test['auroc'].append(test_results['auroc'])
        res_dict_test['auprc'].append(test_results['auprc'])
        res_dict_test['best_f1'].append(test_results['best_f1'])
        res_dict_test['loss'].append(test_results['loss'])

    # print(f"[{'K-fold CV'}] name:{args.exp_name}, pred_offset:{args.pred_before}, "
    logger.info(f"[{'K-fold CV'}] name:{args.exp_name}, pred_offset:{args.pred_before}, "
                f"task:{args.task_idx}, "
                f"avg. loss: {np.array(res_dict_val['loss']).mean():.4f}, "
                f"avg. AUROC: {np.array(res_dict_val['auroc']).mean():.4f}, "
                f"avg. AUPRC: {np.array(res_dict_val['auprc']).mean():.4f}, "
                f"avg. F1: {np.array(res_dict_val['best_f1']).mean():.4f} "
                )
    logger.info(f"[{'Avg score on the Test set'}] name:{args.exp_name}, pred_offset:{args.pred_before}, "
                f"task:{args.task_idx}, "
                f"avg. loss: {np.array(res_dict_test['loss']).mean():.4f}, "
                f"avg. AUROC: {np.array(res_dict_test['auroc']).mean():.4f}, "
                f"avg. AUPRC: {np.array(res_dict_test['auprc']).mean():.4f}, "
                f"avg. F1: {np.array(res_dict_test['best_f1']).mean():.4f} "
                )
