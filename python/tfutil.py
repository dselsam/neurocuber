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
from collections import namedtuple

def decode_transfer_fn(transfer_fn):
    if transfer_fn == "relu": return tf.nn.relu
    elif transfer_fn == "tanh": return tf.nn.tanh
    elif transfer_fn == "sig": return tf.nn.sigmoid
    elif transfer_fn == "elu": return tf.nn.elu
    else:
        raise Exception("Unsupported transfer function %s" % transfer_fn)

def repeat_end(val, n, k):
    return [val for i in range(n)] + [k]

def build_l2_cost():
    l2_cost = tf.zeros([])
    for var in tf.trainable_variables():
        l2_cost += tf.nn.l2_loss(var)
    return l2_cost

def build_learning_rate(cfg, global_step):
    lr = cfg['learning_rate']
    if lr['kind'] == "none":
        return tf.constant(lr['start'])
    elif lr['kind'] == "poly":
        return tf.train.polynomial_decay(learning_rate=lr['start'],
                                         global_step=global_step,
                                         end_learning_rate=lr['end'],
                                         decay_steps=lr['decay_steps'],
                                         power=lr['power'])
    elif lr['kind'] == "exp":
        return tf.train.exponential_decay(learning_rate=lr['start'],
                                          global_step=global_step,
                                          decay_steps=lr['decay_steps'],
                                          decay_rate=lr['decay_rate'],
                                          staircase=False)
    else:
        raise Exception("lr_decay_type must be 'none', 'poly' or 'exp'")

def build_apply_gradients(cfg, loss, learning_rate, global_step):
    optimizer            = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _         = tf.clip_by_global_norm(gradients, cfg['clip_val'])
    apply_gradients      = optimizer.apply_gradients(zip(gradients, variables), name='apply_gradients', global_step=global_step)
    return apply_gradients

def summarize_tensor(name, x):
    with tf.name_scope(name):
        mean = tf.reduce_mean(x)
        tf.summary.scalar('min', tf.reduce_min(x))
        tf.summary.scalar('max', tf.reduce_max(x))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(x - mean))))
        tf.summary.histogram('histogram', x)

def mean_batch_norm(x):
    # x : (n, d)
    # (d)
    mu = tf.reduce_mean(x, axis=0)
    # (1, d)
    mu = tf.expand_dims(mu, axis=0)
    # (n, d)
    mu = tf.tile(mu, [tf.shape(x)[0], 1])
    return x - mu
