# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np
import math
import random
from mlp import MLP
from tfutil import repeat_end, decode_transfer_fn, mean_batch_norm
from collections import namedtuple

NeuroSATDatapoint = namedtuple('NeuroSATDatapoint',  ['n_vars', 'n_clauses', 'LC_idxs', 'target_var', 'target_sl_esteps'])

class NeuroSATParameters:
    def __init__(self, cfg):
        if cfg['repeat_layers']:
            self.L_updates = [MLP(cfg, 2 * cfg['d_lit'] + cfg['d_clause'], repeat_end(cfg['d_lit'], cfg['n_update_layers'], cfg['d_lit']),
                                  name="L_u", nl_at_end=cfg['mlp_update_nl_at_end'])] * cfg['n_rounds']
            self.C_updates = [MLP(cfg, cfg['d_lit'] + cfg['d_clause'], repeat_end(cfg['d_clause'], cfg['n_update_layers'], cfg['d_clause']),
                                  name="C_u", nl_at_end=cfg['mlp_update_nl_at_end'])] * cfg['n_rounds']
        else:
            self.L_updates = [MLP(cfg, 2 * cfg['d_lit'] + cfg['d_clause'], repeat_end(cfg['d_lit'], cfg['n_update_layers'], cfg['d_lit']),
                                  name=("L_u_%d" % t), nl_at_end=cfg['mlp_update_nl_at_end']) for t in range(cfg['n_rounds'])]
            self.C_updates = [MLP(cfg, cfg['d_lit'] + cfg['d_clause'], repeat_end(cfg['d_clause'], cfg['n_update_layers'], cfg['d_clause']),
                                  name=("C_update_%d" % t), nl_at_end=cfg['mlp_update_nl_at_end']) for t in range(cfg['n_rounds'])]

        self.L_score = MLP(cfg, 2 * cfg['d_lit'], repeat_end(cfg['d_lit'], cfg['n_score_layers'], 2), name=("L_score"), nl_at_end=False)

NeuroSATArgs = namedtuple('NeuroSATArgs', ['n_vars', 'n_clauses', 'LC_idxs'])

class NeuroSAT(object):
    def __init__(self, cfg, args):
        n_vars, n_lits, n_clauses = args.n_vars, 2 * args.n_vars, args.n_clauses

        L  = tf.ones(shape=[2 * args.n_vars, cfg['d_lit']], dtype=tf.float32)
        C  = tf.ones(shape=[args.n_clauses, cfg['d_clause']], dtype=tf.float32)

        LC = tf.SparseTensor(indices=tf.cast(args.LC_idxs, tf.int64),
                              values=tf.ones(tf.shape(args.LC_idxs)[0]),
                              dense_shape=[tf.cast(n_lits, tf.int64), tf.cast(n_clauses, tf.int64)])

        params = NeuroSATParameters(cfg)

        def flip(lits): return tf.concat([lits[n_vars:, :], lits[0:n_vars, :]], axis=0)

        for t in range(cfg['n_rounds']):
            C_old, L_old = C, L

            LC_msgs = tf.sparse_tensor_dense_matmul(LC, L, adjoint_a=True) * cfg['LC_scale']
            C       = params.C_updates[t].forward(tf.concat([C, LC_msgs], axis=-1))

            if cfg['batch_norm']: C = mean_batch_norm(C)
            if cfg['res_layers']: C = C + C_old

            CL_msgs = tf.sparse_tensor_dense_matmul(LC, C, adjoint_a=False) * cfg['CL_scale']
            L       = params.L_updates[t].forward(tf.concat([L, CL_msgs, flip(L)], axis=-1))

            if cfg['batch_norm']: L = mean_batch_norm(L)
            if cfg['res_layers']: L = L + L_old

        scores = params.L_score.forward(tf.concat([L[0:n_vars, :], L[n_vars:, :]], axis=1))

        self.logits           = scores[:, 0]
        self.sl_esteps_scores = scores[:, 1]
        self.sl_esteps        = tf.reduce_mean(self.sl_esteps_scores, axis=0)
