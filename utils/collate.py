import torch as t
from config import args
import pdb

def collate_fn(batch):
    '''
    batch: (bsz, tuple)
    tuple: (data, label)
    data shape: (?, 51)
    label shape: (?, )
    Puts each data field into a tensor with outer dimension batch size
    '''

    if args.modality == 'emr': # 61
        dim_modal = args.dim_emr_feats + args.dim_pos
    elif args.modality == 'txt': # 800
        dim_modal = args.dim_embed + args.dim_pos
    elif args.modality == 'both': # 829
        dim_modal = args.dim_emr_feats + args.dim_embed + args.dim_pos
    max_num_context = args.max_num_context
    batch_size = len(batch)

    num_context = t.randint(1, max_num_context, size=(1,))[0]
    num_target = t.Tensor([args.num_target]).type(t.int32)[0]

    cx_arr = t.zeros([batch_size, num_context, dim_modal])
    cy_arr = t.zeros([batch_size, num_context, args.dim_output])
    tx_arr = t.zeros([batch_size, num_target, dim_modal])
    ty_arr = t.zeros([batch_size, num_target, args.dim_output])
    pos_c = t.zeros([batch_size, num_context, args.dim_pos])
    pos_t = t.zeros([batch_size, num_target, args.dim_pos])

    if args.is_after_task:
        pred_bf = t.Tensor([args.pred_before]).type(t.int32)[0] # AFTER task
    else:
        pred_bf = t.randint(1, args.pred_before+1, size=(1,))[0] # WITHIN task

    for bidx, (data, label, pos) in enumerate(batch):
        len_lab_m1 = t.Tensor([len(label)-1]).type(t.int32)[0]

        has_onset = t.argmax(label) # find first index from time steps
        if has_onset < 1: # if negative sample or onset at first timestep
            has_onset = t.randint(int(t.minimum(pred_bf, len_lab_m1)), len(label), size=(1,))[0]

        x_base_idx = t.max(has_onset - pred_bf, t.Tensor([1])).type(t.int32)[0] # 2 - 6 = -4, -> 1
        # sample context observations
        if x_base_idx < num_context: # if too short, replacement=True
            c_idx = t.randint(0, int(x_base_idx), size=(num_context,))
        else:
            c_idx = t.randperm(int(x_base_idx))[:num_context] # without replacement
        cx_arr[bidx, :num_context] = data[c_idx] # time steps
        cy_arr[bidx, :num_context, 0] = label[t.minimum(c_idx + pred_bf, has_onset-1)] # cxt + pred_bf steps
        pos_c[bidx, :num_context] = pos[c_idx]

        # target
        tx_arr[bidx, :] = data[x_base_idx] # first onset time
        ty_arr[bidx, :, 0] = label[has_onset] # cxt + pred_bf steps
        pos_t[bidx, :] = pos[x_base_idx]

    return cx_arr, cy_arr, tx_arr, ty_arr, pos_c, pos_t
