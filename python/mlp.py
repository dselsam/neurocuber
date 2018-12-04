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
import numpy as np
import tensorflow as tf
from tfutil import decode_transfer_fn

class MLP(object):
    def __init__(self, cfg, d_in, d_outs, name, nl_at_end=True):
        self.cfg = cfg
        self.name = name
        self.transfer_fn = decode_transfer_fn(cfg['mlp_transfer_fn'])
        self.nl_at_end = nl_at_end
        self._init_weights(d_in, d_outs)

    def _init_weights(self, d_in, d_outs):
        self.ws = []
        self.bs = []

        d = d_in

        with tf.variable_scope(self.name) as scope:
            for i, d_out in enumerate(d_outs):
                with tf.variable_scope('%d' % i) as scope:
                    if self.cfg['weight_reparam']:
                        w = tf.get_variable(name="w", shape=[d, d_out], initializer=tf.contrib.layers.xavier_initializer())
                        g = tf.get_variable(name="g", shape=[1, d_out], initializer=tf.ones_initializer())
                        self.ws.append(tf.nn.l2_normalize(w, axis=0) * tf.tile(g, [d, 1]))
                    else:
                        self.ws.append(tf.get_variable(name="w", shape=[d, d_out], initializer=tf.contrib.layers.xavier_initializer()))

                    self.bs.append(tf.get_variable(name="b", shape=[d_out], initializer=tf.zeros_initializer()))
                d = d_out

    def forward(self, z):
        x = z
        for i in range(len(self.ws)):
            if 'dropout_rate' in self.cfg and self.cfg['dropout_rate'] > 0.0:
                x = tf.layers.dropout(x, rate=self.cfg['dropout_rate'], training=self.cfg['dropout_training'])
            x = tf.matmul(x, self.ws[i]) + self.bs[i]
            if self.nl_at_end or i + 1 < len(self.ws):
                x = self.transfer_fn(x)
        return x
