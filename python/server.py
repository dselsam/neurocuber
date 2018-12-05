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
import subprocess
import random
import sys
import math
import os
import time
from learner import Learner
from actor import ActorEpisodeResult
import queue
from util import set_pyro_config
from tbwriter import TensorBoardWriter
from sat_util import parse_dimacs
from replay_buffer import ReplayBuffer
import threading
import Pyro4

class LearnerThread(threading.Thread):
    def __init__(self, learner):
        threading.Thread.__init__(self)
        self.learner = learner

    def run(self):
        while True:
            self.learner.step()

class LearnerPostThread(threading.Thread):
    def __init__(self, cfg, learner_outqueue, tbwriter):
        threading.Thread.__init__(self)
        self.cfg = cfg
        self.learner_outqueue = learner_outqueue
        self.tbwriter = tbwriter

    def run(self):
        while True:
            self.tbwriter.log_learner_episode(*self.learner_outqueue.get())

@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class NeuroCuberServer:
    def __init__(self, cfg):
        self.cfg                 = cfg
        self.tbwriter            = TensorBoardWriter(cfg)
        self.replay_buffer       = ReplayBuffer(cfg)
        self.learner_outqueue    = queue.Queue()
        self.learner             = Learner(cfg, replay_buffer=self.replay_buffer, outqueue=self.learner_outqueue)
        self.learner_thread      = LearnerThread(self.learner)
        self.learner_thread.start()

        self.learner_post_thread = LearnerPostThread(cfg, self.learner_outqueue, self.tbwriter)
        self.learner_post_thread.start()

    def get_config(self):
        return self.cfg

    def get_weights(self):
        return self.learner.get_weights()

    def process_actor_episode(self, aer):
        aer = ActorEpisodeResult(*aer)
        assert(aer is not None)

        self.tbwriter.log_actor_episode(aer)
        self.replay_buffer.add_datapoints(aer.datapoints)

def get_options():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', action='store', type=str)
    parser.add_argument('--config', action='store', dest='config', type=str, default='config/server.json')
    parser.add_argument('--root_dir', action='store', dest='root_dir', type=str, default='.')
    parser.add_argument('--host', action='store', dest='host', type=str, default="0.0.0.0")
    parser.add_argument('--port', action='store', dest='port', type=int, default=9091)
    return parser.parse_args()

def load_config(opts):
    import json
    import datetime
    with open(opts.config) as f: cfg = json.load(f)

    cfg['experiment']          = opts.experiment
    cfg['datetime']            = "_".join((str(datetime.datetime.now())).split('.')[0].split()).replace(':', '_')
    cfg['commit']              = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
    cfg['hostname']            = subprocess.check_output(['hostname']).strip().decode('ascii')
    cfg['unique_id']           = "%s_%s" % (cfg['experiment'], cfg['datetime'])
    cfg['root_dir']            = opts.root_dir
    cfg['run_dir']             = os.path.join(cfg['root_dir'], "runs", cfg['unique_id'])
    cfg['summary_dir']         = os.path.join(cfg['run_dir'], "summaries")
    cfg['checkpoint_dir']      = os.path.join(cfg['run_dir'], "checkpoints")

    os.makedirs(cfg['run_dir'])
    os.makedirs(cfg['summary_dir'])
    os.makedirs(cfg['checkpoint_dir'])

    with open(os.path.join(cfg['run_dir'], 'config.json'), 'w') as f: json.dump(cfg, f)

    return cfg


def main():
    print("---Welcome to AlphaCuber---")
    opts = get_options()
    cfg  = load_config(opts)
    set_pyro_config()
    Pyro4.Daemon.serveSimple({ NeuroCuberServer(cfg): "neurocuber_server" }, host=opts.host, port=opts.port, ns=False)

if __name__ == "__main__":
    main()
