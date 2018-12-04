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
import sqlite3
import subprocess
import math
from client import ActorEpisodeResult
import os
import tensorflow as tf
import numpy as np
import collections

class TensorBoardWriter:
    def __init__(self, cfg):
        self.cfg     = cfg
        subprocess.run(["killall", "tensorboard"])
        subprocess.Popen(["tensorboard", "--logdir", self.cfg['summary_dir'], "--host", "0.0.0.0", "--port", "6006"])

        self.writers = { "learn" : tf.summary.FileWriter(os.path.join(cfg['summary_dir'], "learn")) }
        self.actor_iterations  = collections.defaultdict(int)

    def flush(self):
        for writer in self.writers:
            writer.flush()

    def log_actor_episode(self, actor_episode_result):
        aer = actor_episode_result

        if aer.cuber not in self.writers:
            self.writers[aer.cuber] = tf.summary.FileWriter(os.path.join(self.cfg['summary_dir'], aer.cuber))

        self.actor_iterations[(aer.cuber, aer.brancher)] += 1

        self.writers[aer.cuber].add_summary(
            tf.Summary(value=[tf.Summary.Value(tag="estimate/vs_%s" % aer.brancher, simple_value=aer.estimate)]),
            global_step=self.actor_iterations[(aer.cuber, aer.brancher)]
        )

    def log_learner_episode(self, iteration, summary, n_secs):
        self.writers['learn'].add_summary(summary, global_step=iteration)
        self.writers['learn'].add_summary(tf.Summary(value=[tf.Summary.Value(tag="profile/n_secs_learn", simple_value=n_secs)]),
                                          global_step=iteration)
