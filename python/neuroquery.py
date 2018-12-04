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
import os
import queue
from neurosat import NeuroSAT, NeuroSATArgs
from sat_util import Var, parse_dimacs

class NeuroQuery:
    def __init__(self, cfg, gpu_id, gpu_frac):
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.queue = queue.Queue()

        tfconfig  = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        self.sess = tf.Session(config=tfconfig)

        self.n_vars     = tf.placeholder(dtype=tf.int32, shape=[], name="Placeholder_n_vars")
        self.n_clauses  = tf.placeholder(dtype=tf.int32, shape=[], name="Placeholder_n_clauses")
        self.LC_idxs    = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="Placeholder_LC_idxs")

        self.neurosat = NeuroSAT(cfg, NeuroSATArgs(n_vars=self.n_vars, n_clauses=self.n_clauses, LC_idxs=self.LC_idxs))

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.assign_placeholders = { tvar.name : tf.placeholder(tvar.value().dtype, tvar.get_shape().as_list(), name="Placeholder_%s" % tvar.name.split(":")[0]) for tvar in tvars }
        self.assign_ops          = [ tvar.assign(self.assign_placeholders[tvar.name]) for tvar in tvars ]

    def query(self, n_vars, n_clauses, LC_idxs):
        logits, v = self.sess.run([self.neurosat.logits, self.neurosat.v], feed_dict={ self.n_vars:n_vars, self.n_clauses:n_clauses, self.LC_idxs:LC_idxs })
        return {"logits":logits, "v":v}

    def restore(self, restore_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, restore_path)

    def set_weights(self, weights):
        names, values = weights
        self.sess.run(self.assign_ops, feed_dict={ self.assign_placeholders[name] : value for (name, value) in zip(names, values) })
