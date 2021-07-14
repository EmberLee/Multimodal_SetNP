import pdb
import os
import pickle as pkl
from tqdm import tqdm
# files
from config import args
from utils.misc import *
from data_loader import get_Modal_loader
from evaluate import evaluate
from models.Soobin.network import LatentModel as ANP_Model
from models.Dupont.network import NeuralProcess
from models.Set_NP.network import SetNP_ViT, SetNP
from baselines.LSTM.network import SimpleLSTM
from baselines.torch_BiLSTM.network import BiLSTM
from baselines.torch_BiLSTM.network import BiLSTM_Att

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from sklearn.model_selection import StratifiedKFold, train_test_split


def train(dataloader, val_loader, model, device, logger, task_idx, fold_idx=0):

    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model.to(device), device_ids=range(args.num_gpu)) # for multi-gpu
        model.to(device)
        cudnn.benchmark = False

    loss_fn = nn.BCELoss() # loss for validation


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0,
                                                            # T_mult=args.t_mult, eta_min=args.eta_min)

    best_val = 0.0
    patience = 0
    start_epoch = 0

    if args.resume:
        ckpt, best_val, start_epoch = load_checkpoint(os.path.join(args.save_path, args.exp_name), model, optimizer=optimizer, is_best=not args.is_last)

        # set lr from paused point - it has not been implemented yet.
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, args.base_lr, args.learning_rate,
                        step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0,
                            scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                            last_epoch=-1, verbose=False)
                            # last_epoch=start_epoch*len(train_loader), verbose=False)
    else:
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, args.base_lr, args.learning_rate,
                        step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0,
                            scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                            last_epoch=-1, verbose=False)


    for epoch in range(start_epoch, args.max_epoch):
        # Train here
        model.train() # DO NOT REMOVE. return to train mode after evaluate()

        loss_avg = RunningAverage()

        with tqdm(total=len(dataloader)) as tq:
            for i, data in enumerate(dataloader):
                if args.is_imputed:
                    xs, labels, seqlens = data
                    if device.type == 'cuda':
                        xs = xs.to(device)
                        labels = labels.to(device)
                    # pdb.set_trace()
                    if args.model_type.upper() in ['LSTM','BILSTM','BILSTM_ATT']:
                        y_pred = model(xs, seqlens)

                    loss = loss_fn(y_pred, labels)


                else:
                    context_x, context_y, target_x, target_y, pos_c, pos_t = data
                    if device.type == 'cuda':
                        context_x = context_x.to(device)
                        context_y = context_y.to(device)
                        target_x = target_x.to(device)
                        target_y = target_y.to(device)

                    # pass through the latent model
                    if args.model_type.upper() == 'SOOBIN':
                        y_pred, kl, loss = model(context_x, context_y, target_x, target_y)
                    elif args.model_type.upper() == 'DUPONT':
                        y_pred_dist, y_pred, y_pred_sigma, loss = model(context_x, context_y, target_x, target_y)
                    elif args.model_type.upper() == 'SETNP_VIT':
                        y_pred, kl, loss = model(context_x, context_y, target_x, target_y)
                    elif args.model_type.upper() == 'SETNP':
                        y_pred, kl, loss = model(pos_c, pos_t, context_x, context_y, target_x, target_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                updated_lr = scheduler.get_last_lr()[0]

                loss_avg.update(loss.item())


                tq.set_postfix(fold=f'fold{fold_idx}', name=f'{args.exp_name}', epoch=f'{epoch}', lr=f'{updated_lr:.4f}', loss='{:05.3f}'.format(loss_avg()), patience=f"{patience}/{args.max_patience}", pred_before=f'{args.pred_before}')
                tq.update()

        logger.debug(f"[fold{fold_idx} 'train'] name:{args.exp_name}, pred_offset:{args.pred_before}, "
        f"patience:{patience}/{args.max_patience}, "
        f"task:{task_idx}, "
        f"epoch: {epoch}, loss: {loss_avg():.4f}, "
        )

        results, logger = evaluate(val_loader, model, epoch, patience, loss_fn, device, logger, args.task_idx, fold_idx=fold_idx)

        is_best = results['auroc'] > best_val

        save_checkpoint({'epoch': epoch,
                        'score': results['auroc'],
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=os.path.join(args.save_path, args.exp_name, f'fold{fold_idx}'))


        if is_best:
            best_val = results['auroc']
            patience = 0
        else:
            patience += 1

        if patience > args.max_patience:
            logger.info(f"Exceed maximum patience {patience}/{args.max_patience}, early stopping.")
            break

    return logger

def filter_too_short(data, obs_key='EMR_VALID_STEPS'):
    res = {}
    for k, v in data.items():
        onset_idx = np.argmax(v['LABEL'][v[obs_key]][:,0])
        if onset_idx > 0: # if onset case
            if onset_idx <= args.pred_before: # if too short
                continue
        elif len(v['LABEL'][v[obs_key]]) -1 <= args.pred_before:
            continue
        res[k] = v

    return res

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.pilot_activate:
        args.max_epoch = 2
        args.max_patience = 2
        # args.dataset_path = './data/MIMIC/mmnp_xypair/multimodal_MIMIC_mmnp_xypair_bert_sample.pkl'
        args.dataset_path = './data/MIMIC/mmnp_xypair/multimodal_MIMIC_mmnp_xypair_imputed_sample.pkl'

    print("\nTrain the model..")
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

    for fidx, (train_idx, val_idx) in enumerate(skf.split(data_tr, labels_tr)):
        # print(f"TEST: {val_idx}\n\n")
        train_loader, dim_modal = get_Modal_loader(args, data_tr[train_idx], balancing=True)
        val_loader, _ = get_Modal_loader(args, data_tr[val_idx])

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

        logger = train(train_loader, val_loader, model, device, logger, args.task_idx, fold_idx=fidx)
