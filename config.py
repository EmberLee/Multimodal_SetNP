import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', default=False, action='store_true')
parser.add_argument('--gpu_id', type=str, default='5') # '5, 7'
parser.add_argument('--num_gpu', type=int, default=1)

parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--is_last', default=False, action='store_true')
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--dataset_path', type=str, default='./data/MIMIC/mmnp_xypair/multimodal_MIMIC_mmnp_xypair.pkl')

parser.add_argument('--save_path', type=str, default='./exp/')
parser.add_argument('--load_path', type=str, default='./exp/pilot')


# exp
# pilot activate -> epoch, patience = 2
parser.add_argument('--setnp_test', default=False, action='store_true')
parser.add_argument('--pilot_activate', default=False, action='store_true')
parser.add_argument('--exp_name', type=str, default='pilot')
parser.add_argument('--seed', type=int, default=1004)
parser.add_argument('--is_after_task', default=False, action='store_true', help='prediction task. WITHIN or AFTER task.')

parser.add_argument('--num_kfold', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--use_clr', default=False, action='store_true', help='use cyclic learning rate')
parser.add_argument('--base_lr', type=float, default=1e-5, help='lower bound of lr when use CLR')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--drop_rate', type=float, default=0.0)

parser.add_argument('--max_epoch', type=int, default=2000)
parser.add_argument('--max_patience', type=int, default=30)


# data
parser.add_argument('--task_idx', type=int, default=0) # 0 for sep, 1 for mortality
parser.add_argument('--pred_before', type=int, default=6)
parser.add_argument('--exact_before', default=False, action='store_true')
parser.add_argument('--max_num_context', type=int, default=10)
parser.add_argument('--num_target', type=int, default=1)
parser.add_argument('--is_imputed', default=False, action='store_true')
parser.add_argument('--max_los', type=int, default=72)


# model
parser.add_argument('--model_type', type=str, default='soobin', help='soobin, dupont, set_soobin') # 29(emr feat) + 32(pos) = 61
parser.add_argument('--modality', type=str, default='both', help='choose certain modality to load.')
# parser.add_argument('--dim_emr', type=int, default=61) # 29(emr feat) + 32(pos) = 61
parser.add_argument('--dim_emr_feats', type=int, default=29) # 29(emr feat) + 32(pos) = 61
parser.add_argument('--dim_hidden', type=int, default=32)
parser.add_argument('--dim_output', type=int, default=1)
parser.add_argument('--dim_pos', type=int, default=32)
parser.add_argument('--dim_set', type=int, default=64) # must be divisible by num_heads
parser.add_argument('--num_heads_pma', type=int, default=4)


# model-text
parser.add_argument('--model_path', type=str, default='dmis-lab/biobert-base-cased-v1.1')
parser.add_argument('--embedding_path', type=str, default='data/embeddings.pkl')
parser.add_argument('--use_embedding_file', default=False, action='store_true',
                    help='use xypair.pkl & embedding.pkl or use xypair_entire.pkl')
parser.add_argument('--dim_embed', type=int, default=768)
parser.add_argument('--dim_txt', type=int, default=800) # 768(txt embed dim) + 32(pos) = 61
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--pooling', type=str, default='mean',
                    help='how to pool word vectors in a sentence.')
parser.add_argument('--split', type=str, default='val')

# baseline
parser.add_argument('--lstm_layers', type=int, default=1)

args = parser.parse_args()
