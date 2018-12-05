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
import math
import random
import time
import util
from neurosat import NeuroSATDatapoint
from neuroquery import NeuroQuery
from sat_util import *
from collections import namedtuple
from actor import mk_actor, mk_cuber, mk_brancher

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', dest='config', type=str, default="config/client.json")
    parser.add_argument('--uri', action='store', dest='uri', type=str, default='PYRO:neurocuber_server@0.0.0.0:9091')
    opts = parser.parse_args()
    print("Options:", opts)

    import json
    with open(opts.config) as f: client_cfg = json.load(f)
    print("Client config:", client_cfg)

    import Pyro4

    from util import set_pyro_config
    set_pyro_config()

    import sys
    sys.excepthook = Pyro4.util.excepthook

    def construct_proxy_server():
        # TODO(dselsam): probably need to help it find the NS
        if opts.uri is None:
            with Pyro4.locateNS() as ns:
                return Pyro4.Proxy(ns.locate(client_cfg['server_name']))
        else:
            return Pyro4.Proxy(opts.uri)

    n_actors_total = sum([actor['n'] for actor in client_cfg['actors']])
    gpu_frac = client_cfg['gpu_frac'] * client_cfg['n_gpus'] / n_actors_total

    def get_actor_info(idx):
        i = 0
        for actor_info in client_cfg['actors']:
            i += actor_info['n']
            if idx < i:
                return actor_info
        raise Exception("could not find actor info for %d" % idx)

    def launch_actor(actor_idx):
        server = construct_proxy_server()
        gpu_id = actor_idx % client_cfg['n_gpus'] if client_cfg['n_gpus'] > 0 else 0
        actor  = mk_actor(server, gpu_id, gpu_frac, get_actor_info(actor_idx))
        actor.loop()

    import multiprocessing
    actors = []
    print("Launching actors...")
    for actor_idx in range(n_actors_total):
        actor = multiprocessing.Process(target=launch_actor, args=(actor_idx,))
        actor.start()
        actors.append(actor)

    print("All actors launched.")
    for actor in actors:
        actor.join()
