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
import threading
import subprocess
import random
import math
import os
import time
import queue
import copy
from neurosat import NeuroSAT, NeuroSATArgs
from tfutil import build_l2_cost, build_learning_rate, build_apply_gradients, summarize_tensor

class Learner:
    def __init__(self, cfg, replay_buffer, outqueue):
        self.cfg            = cfg
        self.replay_buffer  = replay_buffer
        self.outqueue       = outqueue

        config    = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        tf.set_random_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

        def get_next_datapoint():
            while True:
                datapoints = self.replay_buffer.sample_datapoints(n_samples=1)
                if not datapoints:
                    print("[LEARNER:GENERATOR] going to sleep...")
                    time.sleep(20)
                    print("[LEARNER:GENERATOR] waking up...")
                else:
                    assert(len(datapoints) == 1)
                    dp = datapoints[0]
                    yield dp.n_vars, dp.n_clauses, dp.LC_idxs, dp.target_var, dp.target_sl_esteps

        dataset = tf.data.Dataset.from_generator(
            get_next_datapoint,
            (tf.int32, tf.int32, tf.int32, tf.int32, tf.float32),
            (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None, 2]), tf.TensorShape([]), tf.TensorShape([]))
        )

        dataset = dataset.prefetch(cfg['prefetch'])
        (n_vars, n_clauses, LC_idxs, target_var, target_sl_esteps) = dataset.make_one_shot_iterator().get_next()

        self.neurosat        = NeuroSAT(cfg, NeuroSATArgs(n_vars=n_vars, n_clauses=n_clauses, LC_idxs=LC_idxs))
        self.p_cost          = cfg['p_cost_scale'] * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.neurosat.logits, labels=target_var)
        self.v_cost          = cfg['v_cost_scale'] * tf.square(self.neurosat.sl_esteps - target_sl_esteps)
        self.l2_cost         = cfg['l2_cost_scale'] * build_l2_cost()
        self.loss            = self.p_cost + self.v_cost + self.l2_cost
        self.global_step     = tf.get_variable("global_step", shape=[], initializer=tf.zeros_initializer(), trainable=False)
        self.learning_rate   = build_learning_rate(cfg, self.global_step)
        self.apply_gradients = build_apply_gradients(cfg, self.loss, self.learning_rate, self.global_step)

        self.target_sl_esteps = target_sl_esteps
        self.declare_summaries()

        tf.global_variables_initializer().run(session=self.sess)
        self.saver = tf.train.Saver(max_to_keep=cfg['max_saves_to_keep'])

        if cfg['restore_path'] != "none":
            print("Restoring from %s..." % cfg['restore_path'])
            self.saver.restore(self.sess, cfg['restore_path'])

        self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.weights = self._extract_weights()

    def declare_summaries(self):
        with tf.name_scope('learn'):
            tf.summary.scalar('total_cost', self.loss)
            tf.summary.scalar('p_cost', self.p_cost)
            tf.summary.scalar('v_cost', self.v_cost)
            tf.summary.scalar('l2_cost', self.l2_cost)
            tf.summary.scalar('target_sl_esteps', self.target_sl_esteps)
            tf.summary.scalar('learning_rate', self.learning_rate)

        self.summary = tf.summary.merge_all()

    def save_checkpoint(self, iteration):
        self.saver.save(self.sess, os.path.join(self.cfg['checkpoint_dir'], "checkpoint"), global_step=int(iteration))

    def _extract_weights(self):
        names = [tvar.name for tvar in self.tvars]
        vals  = self.sess.run(self.tvars)
        return names, vals

    def get_weights(self):
        return self.weights

    def step(self):
        start = time.time()
        _, iteration, summary = self.sess.run([self.apply_gradients, self.global_step, self.summary])
        end   = time.time()
        self.outqueue.put((iteration, summary, end - start))

        if iteration % self.cfg['update_weights_freq'] == 0:
            self.weights = self._extract_weights()

        if iteration % self.cfg['checkpoint_freq'] == 0:
            self.save_checkpoint(iteration)
