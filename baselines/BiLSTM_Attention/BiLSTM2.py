# coding: utf-8

import os
import csv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib import rnn
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# from common.param.model_param import *
# from common.model_class.Model import Model
# from common.model_class.Feature import DYNAMIC_FEATURES
# from common.util import model_calibration as calib
# from common.util import clr as CLR

from baselines.BiLSTM_Attention.model_param import *
from baselines.BiLSTM_Attention import Model
from baselines.BiLSTM_Attention import DYNAMIC_FEATURES
from baselines.BiLSTM_Attention import clr as CLR

N_UNIQUE_DYNAMIC_FEATURES = 18

def expanded_pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
         differences = x.unsqueeze(1) - y.unsqueeze(0) # N,1,d - 1,M,d
    else:
        differences = tf.expand_dims(x,1) - tf.expand_dims(x,0)
    distances = tf.reduce_sum(differences * differences, -1)
    return distances


def LDA_Loss(embedding, label, cv_type=1):

    pos_idx = tf.where(tf.equal(label,1.0))
    neg_idx = tf.where(tf.equal(label,0.0))

    pos_idx = pos_idx[:,0]
    neg_idx = neg_idx[:,0]

    # tf.gather: indices.shape + params.shape[1:] (params, indices)
    pos_embedding = tf.gather(embedding, pos_idx)
    neg_embedding = tf.gather(embedding, neg_idx)

    pos_distance = tf.reduce_mean(expanded_pairwise_distances(pos_embedding))
    neg_distance = tf.reduce_mean(expanded_pairwise_distances(neg_embedding))

    return pos_distance + neg_distance


class BiLSTM2(Model):
    def __init__(self, arg):
        super().__init__(arg)

        if arg.ward == 'GW':
            MAX_SEQ_LENGTH = 24 * 7 + 1
        # elif arg.event == 'DEATH':
        #     MAX_SEQ_LENGTH = 24 * 7 + 1
        else:
            MAX_SEQ_LENGTH = 70 + 1

        if arg.event == 'PTE':
            N_UNIQUE_DYNAMIC_FEATURES = 19
        else:
            N_UNIQUE_DYNAMIC_FEATURES = 18

        self.model_dir_base = 'model-{}'.format(self.arg.exp_id)
        self.model_dir = os.path.join(self.arg.train_dir, self.model_dir_base)

        self.lambda_reg = 0.01
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.lambda_reg)

        self.n_labels = 1

        print('Build Model...')

        with tf.name_scope('ph') as scope:
            self.X_static = tf.compat.v1.placeholder(tf.float32, [None, N_UNIQUE_DYNAMIC_FEATURES+1], name='xs_static')
            if self.arg.use_delta:
                self.X_dynamic = tf.compat.v1.placeholder(tf.float32, [None, MAX_SEQ_LENGTH, N_UNIQUE_DYNAMIC_FEATURES*2], name='xs_dynamic')
            else:
                self.X_dynamic = tf.compat.v1.placeholder(tf.float32, [None, MAX_SEQ_LENGTH, N_UNIQUE_DYNAMIC_FEATURES], name='xs_dynamic')

            self.X_seq_len = tf.compat.v1.placeholder(tf.int32, [None], name='xs_seq_len')

            self.Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ys')
            # self.lda_on = tf.compat.v1.placeholder(tf.float32, [], name='lda_on')

            self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
            self.drop_rate = tf.compat.v1.placeholder(tf.float32, name='drop_rate')
            self.lr_keep = tf.compat.v1.placeholder(tf.float32, name='lr_keep')
            self.epoch = tf.compat.v1.placeholder(tf.int32, name='epoch')
            print('X_static', self.X_static.shape)
            print('X_dynamic', self.X_dynamic.shape)
            print('X_seq_len', self.X_seq_len.shape)
            print('Y', self.Y.shape)

            self.batch_size = tf.shape(self.X_static)[0]

        with tf.name_scope('model/static') as scope:
            # net_static & latest
            print('net_static/shape', self.X_static.shape)

            self.static_input = tf.reshape(self.X_static, [-1, 1, 1, N_UNIQUE_DYNAMIC_FEATURES + 1])

            # self.net_static0 = tf.layers.dense(inputs=self.X_static,
            #                       units=32,
            #                       activation=None,
            #                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            #                       kernel_regularizer=self.regularizer)
            # self.net_static0 = tf.contrib.layers.batch_norm(self.net_static0, center=True, scale=True, is_training=self.is_training)
            # self.net_static0 = tf.nn.relu(self.net_static0)

            # print('net/dense_static', self.net_static0.shape)

            # self.net_static1 = tf.layers.dense(inputs=self.net_static0,
            #                       units=32,
            #                       activation=None,
            #                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            #                       kernel_regularizer=self.regularizer)
            # self.net_static1 = tf.contrib.layers.batch_norm(self.net_static1, center=True, scale=None, is_training=self.is_training)
            # self.net_static1 = tf.nn.relu(self.net_static1)

            # print('net/dense_static', self.net_static1.shape)

            self.net_static0 = tf.layers.conv2d(inputs=self.static_input,
                                          filters=16,
                                          kernel_size=[1, 1],
                                          padding='VALID',
                                          data_format='channels_last',
                                          kernel_regularizer=self.regularizer,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                          activation=tf.nn.relu)
            print('net_static/conv0', self.net_static0.shape)

            self.net_static1 = tf.layers.conv2d(inputs=self.net_static0,
                                          filters=16,
                                          kernel_size=[1, 1],
                                          padding='VALID',
                                          data_format='channels_last',
                                          kernel_regularizer=self.regularizer,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                          activation=tf.nn.relu)
            print('net_static/conv1', self.net_static1.shape)

            self.net_static1 = tf.layers.flatten(self.net_static1)
            print('net_static/flatten', self.net_static1.shape)

        with tf.name_scope('model/dynamic') as scope:
            print('net_dynamic/dynamic_input', self.X_dynamic.shape)

            if self.arg.lstm_type == 'basic':
                fw_cells = [rnn.BasicLSTMCell(self.arg.z_dim) for _ in range(self.arg.lstm_layers)]
                bw_cells = [rnn.BasicLSTMCell(self.arg.z_dim) for _ in range(self.arg.lstm_layers)]
            elif self.arg.lstm_type == 'layer_norm':
                fw_cells = [rnn.LayerNormBasicLSTMCell(num_units=self.arg.z_dim) for _ in range(self.arg.lstm_layers)]
                bw_cells = [rnn.LayerNormBasicLSTMCell(num_units=self.arg.z_dim) for _ in range(self.arg.lstm_layers)]

            # fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=(1.0 - self.drop_rate)) for cell in fw_cells]
            # bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=(1.0 - self.drop_rate)) for cell in bw_cells]

            self.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.X_dynamic, sequence_length=self.X_seq_len, dtype=tf.float32)

        with tf.name_scope("attention"):
            self.attention_score = tf.nn.softmax(tf.layers.dense(self.rnn_outputs, 1, activation=tf.nn.tanh), axis=1)
            self.attention_out = tf.squeeze(
                tf.matmul(tf.transpose(self.rnn_outputs, perm=[0, 2, 1]), self.attention_score),
                axis=-1)

            print('net/attention_out', self.attention_out.shape)


        with tf.name_scope('model/concat') as scope:
            self.net_concat0 = tf.concat([self.net_static1, self.attention_out], 1)

            print('net/concat', self.net_concat0.shape)

            if self.arg.classifier == 'cnn':

                self.concat_input = tf.reshape(self.net_concat0, [-1, 1, 1, self.net_concat0.shape[1]])

                print('net/concat_input', self.concat_input.shape)

                if self.arg.cnn_layers > 0:
                    self.net_concat1 = tf.layers.conv2d(inputs=self.concat_input,
                                        filters=64,
                                        kernel_size=[1, 1],
                                        padding='VALID',
                                        data_format='channels_last',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.relu)
                    self.net_concat1 = tf.nn.dropout(self.net_concat1, keep_prob = (1.0 - self.drop_rate),
                                    noise_shape = [tf.shape(self.net_concat1)[0], 1, 1, tf.shape(self.net_concat1)[3]])


                    print('net/concat1', self.net_concat1.shape)

                if self.arg.cnn_layers > 1:
                    self.net_concat2 = tf.layers.conv2d(inputs=self.net_concat1,
                                        filters=32,
                                        kernel_size=[1, 1],
                                        padding='VALID',
                                        data_format='channels_last',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.relu)
                    self.net_concat2 = tf.nn.dropout(self.net_concat2, keep_prob = (1.0 - self.drop_rate),
                                    noise_shape = [tf.shape(self.net_concat2)[0], 1, 1, tf.shape(self.net_concat2)[3]])

                    print('net/concat2', self.net_concat2.shape)

                if self.arg.cnn_layers > 2:
                    self.net_concat3 = tf.layers.conv2d(inputs=self.net_concat2,
                                        filters=32,
                                        kernel_size=[1, 1],
                                        padding='VALID',
                                        data_format='channels_last',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.relu)

                    print('net/concat3', self.net_concat3.shape)

                if self.arg.cnn_layers > 3:
                    self.net_concat4 = tf.layers.conv2d(inputs=self.net_concat3,
                                        filters=16,
                                        kernel_size=[1, 1],
                                        padding='VALID',
                                        data_format='channels_last',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.relu)

                    print('net/concat4', self.net_concat4.shape)

                if self.arg.cnn_layers > 4:
                    self.net_concat5 = tf.layers.conv2d(inputs=self.net_concat4,
                                        filters=16,
                                        kernel_size=[1, 1],
                                        padding='VALID',
                                        data_format='channels_last',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.relu)

                    print('net/concat5', self.net_concat5.shape)

                if self.arg.cnn_layers > 5:
                    self.net_concat6 = tf.layers.conv2d(inputs=self.net_concat5,
                                        filters=16,
                                        kernel_size=[1, 1],
                                        padding='VALID',
                                        data_format='channels_last',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.relu)

                    print('net/concat6', self.net_concat6.shape)

                if self.arg.cnn_layers == 1:
                    self.concat_output = tf.layers.flatten(self.net_concat1)
                elif self.arg.cnn_layers == 2:
                    self.concat_output = tf.layers.flatten(self.net_concat2)
                elif self.arg.cnn_layers == 3:
                    self.concat_output = tf.layers.flatten(self.net_concat3)
                elif self.arg.cnn_layers == 4:
                    self.concat_output = tf.layers.flatten(self.net_concat4)
                elif self.arg.cnn_layers == 5:
                    self.concat_output = tf.layers.flatten(self.net_concat5)
                elif self.arg.cnn_layers == 6:
                    self.concat_output = tf.layers.flatten(self.net_concat6)

            elif self.arg.classifier == 'dense':
                self.net_concat1 = tf.layers.dense(inputs=self.net_concat0,
                                  units=256,
                                  activation=None,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                  kernel_regularizer=self.regularizer)
                self.net_concat1 = tf.nn.relu(self.net_concat1)
                self.net_concat1 = tf.layers.dropout(inputs=self.net_concat1,
                                    training=self.is_training,
                                    rate=self.drop_rate)

                print('net/dense_concat1', self.net_concat1.shape)

                self.net_concat2 = tf.layers.dense(inputs=self.net_concat1,
                                      units=256,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      kernel_regularizer=self.regularizer)
                self.net_concat2 = tf.nn.relu(self.net_concat2)
                self.net_concat2 = tf.layers.dropout(inputs=self.net_concat2,
                                    training=self.is_training,
                                    rate=self.drop_rate)

                print('net/dense_concat2', self.net_concat2.shape)

                self.net_concat3 = tf.layers.dense(inputs=self.net_concat2,
                                      units=128,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      kernel_regularizer=self.regularizer)
                self.net_concat3 = tf.nn.relu(self.net_concat3)
                self.net_concat3 = tf.layers.dropout(inputs=self.net_concat3,
                                    training=self.is_training,
                                    rate=self.drop_rate)

                print('net/dense_concat3', self.net_concat3.shape)

                self.net_concat4 = tf.layers.dense(inputs=self.net_concat3,
                                      units=128,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      kernel_regularizer=self.regularizer)
                self.net_concat4 = tf.nn.relu(self.net_concat4)
                self.net_concat4 = tf.layers.dropout(inputs=self.net_concat4,
                                    training=self.is_training,
                                    rate=self.drop_rate)

                print('net/dense_concat4', self.net_concat4.shape)

                self.net_concat5 = tf.layers.dense(inputs=self.net_concat4,
                                      units=64,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      kernel_regularizer=self.regularizer)
                self.net_concat5 = tf.nn.relu(self.net_concat5)
                self.net_concat5 = tf.layers.dropout(inputs=self.net_concat5,
                                    training=self.is_training,
                                    rate=self.drop_rate)

                print('net/dense_concat5', self.net_concat5.shape)

                self.concat_output = self.net_concat5


            print('net/concat_output', self.concat_output.shape)

            self.logit = tf.layers.dense(inputs=self.concat_output,
                                  units=self.n_labels,
                                  activation=None,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='logit')
            print('net/output', self.logit.shape)

            self.prob = tf.nn.sigmoid(self.logit, name='prob')
            self.prediction = tf.cast(self.prob > 0.5, dtype=tf.float32, name='prediction')

            ## Normal cross entropy
            # self.cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logit)

            ## Weighted cross entropy
            self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=self.Y, logits=self.logit, pos_weight=1.5))

            # if self.arg.false_negative:
            #     self.false_negative = tf.cast(((self.Y - self.prediction) > 0), dtype=tf.float32)
            #     self.false_negative *= 10.0
            #     self.false_negative += 1.0
            #     self.cost = self.cost * self.false_negative

            self.cost = tf.reduce_mean(self.cost)

            # if self.arg.lda_on:
            self.Lda_Loss = LDA_Loss(self.concat_output, self.Y, cv_type=1)


            ## Weight Decay Regularization
            if self.arg.kernel_reg:
                print('_____Kernel regualizer ON!_____')
                self.reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.cost += (0.2 * tf.reduce_mean(self.reg))

            self.lda_cost = self.cost + (0.5 * self.Lda_Loss)

            if self.arg.clr_on:
                self.clr = CLR.cyclic_learning_rate(self.epoch, learning_rate=self.arg.learning_rate/2, max_lr=0.005,
                                                    step_size=30, gamma=0.99, mode='exp_range') #exp_range
                if self.arg.optim == 'adam':
                    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr_keep)
                elif self.arg.optim == 'adadelta':
                    self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.lr_keep)
                elif self.arg.optim == 'rmsprop':
                    self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr_keep)
                elif self.arg.optim == 'sgd':
                    self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr_keep)
            else:

                if self.arg.optim == 'adam':
                    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.arg.learning_rate * self.lr_keep)
                elif self.arg.optim == 'adadelta':
                    self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.arg.learning_rate * self.lr_keep)
                elif self.arg.optim == 'rmsprop':
                    # self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.arg.learning_rate * self.lr_keep, decay=0.99, momentum=0.9, centered=True)
                    self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.arg.learning_rate * self.lr_keep)
                elif self.arg.optim == 'sgd':
                    self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.arg.learning_rate * self.lr_keep)

            self.update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_op = self.optimizer.minimize(self.cost)
                self.train_op_lda = self.optimizer.minimize(self.lda_cost)

            self.is_correct = tf.equal(self.prediction, self.Y)
            self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))
