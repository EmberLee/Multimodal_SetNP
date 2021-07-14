# coding: utf-8

import os
import csv
import pickle as pkl

from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from common.param.model_param import *
from common.model_class.Feature import DYNAMIC_FEATURES

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = PER_PROCESS_GPU_MEMORY_FRACTION


OUT_HEADER = [
    'PID',
    'LABEL',
    'PRED',
]


class Model:
    def __init__(self, arg):
        self.arg = arg
        self.exp_title = arg.train_dir.split('/')[9] + ' Exp_{}'.format(arg.exp_id)
        self.log_path = arg.train_dir.split('/')[-4] + '_' + arg.train_dir.split('/')[-2] + '_Exp_{}'.format(arg.exp_id)
        tf.set_random_seed(RANDOM_SEEDS[self.arg.exp_id])

    def _evaluate(self, probs, ys):
        # loss, acc, pred, prob = results

        pred_pos = probs
        y_pos = ys

        roc = roc_auc_score(y_pos, pred_pos)
        pr = average_precision_score(y_pos, pred_pos)

        # return loss, acc, roc, pr
        return roc, pr

    def _print_log(self, eval_results, eval_type, epoch=None):
        # log_fp = open(os.path.join('/mnt/aitrics_ext/ext01/steve/working/VitalCare-Model-v2/script/model/log', self.log_path), 'a')
        log_fp = open(os.path.join('/mnt/aitrics_ext/ext01/lucas/VitalCare-Model-v2/script/model/log', self.log_path), 'a')
        log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log = ['[{}]'.format(log_time)]
        if eval_type in ['train', 'train_calib']:
            log += ['epoch: {:5}'.format(epoch)]

        if eval_type == 'train':
            eval_names = ['<train>', '<valid>']
        elif eval_type == 'train_calib':
            eval_names = ['<train>', '<valid>', '<valid_calib>']
        elif eval_type == 'valid':
            eval_names = ['<valid>']
        elif eval_type == 'valid_calib':
            eval_names = ['<valid_calib>']
        else:
            raise AssertionError()

        for i, eval_result in enumerate(eval_results):
            loss, acc, roc, pr = eval_result
            log += ['{}'.format(eval_names[i]),
                    'loss: {:07.5f}'.format(loss),
                    'acc: {:.3f}'.format(acc),
                    'roc: {:.3f}'.format(roc),
                    'pr: {:.3f}'.format(pr)]

        log = ' '.join(log)
        print(log)
        log_fp.write(log)
        log_fp.write('\n')
        log_fp.close()

    def _write_result(self, valid_ids, valid_ys, valid_result, calibration):
        if calibration:
            result_file_base = 'result-calibration-{}-{}-{}.csv'.format(
                self.arg.train_dataset,
                self.arg.valid_dataset,
                self.arg.exp_id)

            if self.arg.noise_type == 'delayed_input':
                result_file_base = 'Delayed_input_' + result_file_base
            elif self.arg.noise_type == 'add_noise':
                result_file_base = 'Add_noise_' + result_file_base
            elif self.arg.noise_type == 'add_noise_time':
                result_file_base = 'Add_noise_time_' + result_file_base


        else:
            result_file_base = 'result-{}-{}-{}.csv'.format(
                self.arg.train_dataset,
                self.arg.valid_dataset,
                self.arg.exp_id)

            if self.arg.noise_type == 'delayed_input':
                result_file_base = 'Delayed_input_' + result_file_base
            elif self.arg.noise_type == 'add_noise':
                result_file_base = 'Add_noise_' + result_file_base
            elif self.arg.noise_type == 'add_noise_time':
                result_file_base = 'Add_noise_time_' + result_file_base

        result_file = os.path.join(self.arg.valid_dir, result_file_base)

        if self.arg.noise_exp:
            if calibration:
                noise_result_file_base = 'noise_exp-calibration-{}-{}.pkl'.format(
                    self.arg.train_dataset,
                    self.arg.valid_dataset)

                if self.arg.noise_type == 'delayed_input':
                    noise_result_file_base = 'Delayed_input_' + noise_result_file_base
                elif self.arg.noise_type == 'add_noise':
                    noise_result_file_base = 'Add_noise_' + noise_result_file_base
                elif self.arg.noise_type == 'add_noise_time':
                    noise_result_file_base = 'Add_noise_time_' + noise_result_file_base


            else:
                noise_result_file_base = 'noise_exp-{}-{}.pkl'.format(
                    self.arg.train_dataset,
                    self.arg.valid_dataset)

                if self.arg.noise_type == 'delayed_input':
                    noise_result_file_base = 'Delayed_input_' + noise_result_file_base
                elif self.arg.noise_type == 'add_noise':
                    noise_result_file_base = 'Add_noise_' + noise_result_file_base
                elif self.arg.noise_type == 'add_noise_time':
                    noise_result_file_base = 'Add_noise_time_' + noise_result_file_base

            noise_result_file = os.path.join(self.arg.valid_dir, noise_result_file_base)

            if self.arg.noise_type == 'delayed_input' and self.arg.noise_length == 0 and self.arg.exp_id == 0:
                fp = open(noise_result_file, 'wb')
                dic = dict()
            elif self.arg.noise_type == 'add_noise' and self.arg.noise_value == 0 and self.arg.exp_id == 0:
                fp = open(noise_result_file, 'wb')
                dic = dict()
            elif self.arg.noise_type == 'add_noise_time' and self.arg.noise_value == 0 and self.arg.exp_id == 0:
                fp = open(noise_result_file, 'wb')
                dic = dict()
            else:
                with open(noise_result_file, 'rb') as rfp:
                    dic = pkl.load(rfp)
                fp = open(noise_result_file, 'wb')

            roc, pr = self._evaluate(valid_result[3], valid_ys)

            if self.arg.noise_type == 'delayed_input':
                if self.arg.exp_id == 0:
                    dic['ROC_{}'.format(self.arg.noise_length)] = [roc]
                    dic['PR_{}'.format(self.arg.noise_length)] = [pr]
                else:
                    dic['ROC_{}'.format(self.arg.noise_length)].append(roc)
                    dic['PR_{}'.format(self.arg.noise_length)].append(pr)

            elif self.arg.noise_type == 'add_noise':
                if self.arg.exp_id == 0:
                    dic['ROC_{}'.format(self.arg.noise_value)] = [roc]
                    dic['PR_{}'.format(self.arg.noise_value)] = [pr]
                else:
                    dic['ROC_{}'.format(self.arg.noise_value)].append(roc)
                    dic['PR_{}'.format(self.arg.noise_value)].append(pr)

            elif self.arg.noise_type == 'add_noise_time':
                if self.arg.exp_id == 0:
                    dic['ROC_{}'.format(self.arg.noise_value)] = [roc]
                    dic['PR_{}'.format(self.arg.noise_value)] = [pr]
                else:
                    dic['ROC_{}'.format(self.arg.noise_value)].append(roc)
                    dic['PR_{}'.format(self.arg.noise_value)].append(pr)

            pkl.dump(dic, fp)
            fp.close()


        else:
            with open(result_file, 'w', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=OUT_HEADER)
                writer.writeheader()
                for i in range(len(valid_ys)):
                    pid = valid_ids[i]
                    label = int(valid_ys[i])
                    pred_pos = valid_result[3][i]

                    writer.writerow(dict(
                        PID=pid,
                        LABEL='{}'.format(label),
                        PRED='{:.3f}'.format(pred_pos[0]),
                    ))

    def _make_negative_mean_input(self, input_x, feature_set):

        n_dynamic_sequences = self.arg.train_window + 1
        n_static_features = 1
        n_unique_dynamic_features = len(DYNAMIC_FEATURES[self.arg.feature_set])

        vital_num = 7

        X_static = input_x[:, :n_static_features]
        X_dynamic = input_x[:, n_static_features:]

        data = np.copy(X_dynamic)
        data = data.reshape(-1, n_unique_dynamic_features, n_dynamic_sequences)
        data = data.transpose([0,2,1]) # B T E


    ##TODO: Make noised input code
    def _make_noised_input(self, input_x, seq_len):
        noise_type = self.arg.noise_type
        fill_type = self.arg.fill_type
        noise_length = self.arg.noise_length
        noise_value = self.arg.noise_value

        noised_data = []

        vital_num = 7

        data = np.copy(input_x)


        if noise_type == 'delayed_input':
            assert noise_length < data.shape[1] #Noise length should be less than time length

            if noise_length == 0: # Without Noise
                return input_x
            else:
                batch_size = data.shape[0] # batch size
                for i in range(batch_size):
                    _data = data[i]
                    noise_length = int(min(noise_length, seq_len[i]-1))
                    if noise_length < 0:
                        noise_length = 0

                    noised_idx = np.random.choice(vital_num,2,replace=False)
                    if fill_type == 'carry_forward':
                        _data[-(noise_length+1):, noised_idx] = _data[-(noise_length+1), noised_idx]
                    # elif fill_type == 'average':
                    #     average_value = np.mean(_data[noise_length:,:], axis=0)
                    #     #average_value = np.random.rand(11)
                    #     _data[:noise_length, noised_idx] = average_value[noised_idx]
                    #     #_data[0:,:] = average_value[:]
                    else:
                        raise AssertionError()


                    noised_data.append(_data)

                noised_data = np.array(noised_data)

                return noised_data

        elif noise_type == 'add_noise':
            data_shape = data.shape
            data_noise = np.random.normal(0, noise_value, data_shape)


            batch_size = data_shape[0]
            time_size = data_shape[1]
            for i in range(batch_size):
                for j in range(time_size):
                    p = np.random.rand()
                    if p > 0.1 or (seq_len[i] <= j):
                        zero_slice = np.zeros(data_shape[2])
                        data_noise[i][j] = zero_slice

            noised_idx = np.random.choice(vital_num,2,replace=False)
            data[:,:, noised_idx] += data_noise[:,:, noised_idx]

            return data

        elif noise_type == 'add_noise_time':
            if noise_value == 0.0 :
                return input_x

            data_shape = data.shape
            noise = 0.5
            data_noise = np.random.normal(0, noise, data_shape)


            batch_size = data_shape[0]
            time_size = data_shape[1]
            for i in range(batch_size):
                for j in range(time_size):
                    p = np.random.rand()
                    if p > noise_value or (seq_len[i] <= j):
                        zero_slice = np.zeros(data_shape[2])
                        data_noise[i][j] = zero_slice

            data[:,:,:vital_num] += data_noise[:,:,:vital_num]

            return data

    def _write_calib_coef(self, cali_type, coef):
        coef_file = os.path.join(self.arg.valid_dir, '{}.txt'.format(cali_type))
        with open(coef_file, 'w', encoding='utf-8') as f:
            if cali_type == 'temp_scaling':
                f.write('temp\t{}'.format(coef['temp_value'][0]))
            elif cali_type == 'platt_scaling':
                f.write('weight\t{:.f}'.format(coef['weight']))
                f.write('bias\t{}'.format(coef['bias']))
            else:
                assert True
