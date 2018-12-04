import numpy as np
import tensorflow as tf
import os
import math
import random
import time
from neurosat import NeuroSATDatapoint
from neuroquery import NeuroQuery
from sat_util import *
from asat import play_asat, mk_cuber, mk_brancher
from collections import namedtuple

ActorEpisodeResult = namedtuple('ActorEpisodeResult', ['dimacs', 'cuber', 'brancher', 'estimate', 'datapoints'])

class Actor:
    def __init__(self, server, gpu_id, gpu_frac, actor_info):
        self.server     = server
        self.cfg        = server.get_config()
        self.cfg['dropout_training'] = False
        self.neuroquery = NeuroQuery(self.cfg, gpu_id, gpu_frac)

        self.cubers     = [mk_cuber(cuber_info, self.neuroquery) for cuber_info in actor_info['cubers']]
        self.branchers  = [mk_brancher(brancher_info, self.neuroquery) for brancher_info in actor_info['branchers']]
        # TODO(dselsam): if we end up training or testing on a large number of SAT problems, will need to
        # parse the files lazily

        self.sps = []
        for root, subdirs, files in os.walk(actor_info['dimacs_dir']):
            for dimacs in files:
                self.sps.append((dimacs, parse_dimacs(os.path.join(root, dimacs))))

        self.train      = actor_info['train']
        self.z3opts     = Z3Options(max_conflicts=actor_info['solver']['max_conflicts'],
                                    sat_restart_max=actor_info['solver']['sat_restart_max'])

    def loop(self):
        while True:
            (dimacs, sp) = random.choice(self.sps)
            cuber        = random.choice(self.cubers)
            brancher     = random.choice(self.branchers)
            self.play_episode(dimacs, sp, cuber, brancher)

    def play_episode(self, dimacs, sp, cuber, brancher):
        self.neuroquery.set_weights(self.server.get_weights())
        ssat_result = play_asat(self._fresh_solver(sp), cuber, brancher)
        if ssat_result is None: return None

        trail, core, estimate = ssat_result

        datapoints = []
        if self.train:
            min_trail   = [lit for lit in trail if lit in core]
            s           = self._fresh_solver(sp)
            for i in range(len(min_trail)):
                tfq         = s.to_tf_query(assumptions=min_trail[:i])
                if len(tfq.fvars) == 0: break
                target_var  = min_trail[i].var().idx()
                target_v    = self._compute_v(n_steps_left=len(min_trail) - i)
                datapoints.append(NeuroSATDatapoint(n_vars=sp.n_vars(),
                                                    n_clauses=sp.n_clauses(),
                                                    LC_idxs=tfq.LC_idxs,
                                                    target_var=target_var,
                                                    target_v=target_v))

        self.server.process_actor_episode(
            tuple(
                ActorEpisodeResult(
                    dimacs=dimacs,
                    cuber=cuber.name,
                    brancher=brancher.name,
                    estimate=estimate,
                    datapoints=datapoints)
            )
        )

    def _fresh_solver(self, sp):
        return Z3Solver(sp=sp, opts=self.z3opts)

    def _compute_v(self, n_steps_left):
        steps = n_steps_left / self.cfg['v_reward_decay_steps']
        return - self.cfg['v_reward_scale'] * (math.pow(self.cfg['v_reward_base'], steps) - 1)

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
        actor  = Actor(server, gpu_id, gpu_frac, get_actor_info(actor_idx))
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
