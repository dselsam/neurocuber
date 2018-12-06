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

ActorEpisodeResult = namedtuple('ActorEpisodeResult', ['dimacs', 'cuber', 'brancher', 'esteps', 'datapoints'])

def sl_esteps_to_esteps(cfg, sl_esteps):
    return np.power(2, sl_esteps / cfg['log_esteps_scale'])

def esteps_to_sl_esteps(cfg, esteps):
    return np.log2(esteps) * cfg['log_esteps_scale']

def mk_zopts(actor_info):
    return Z3Options(max_conflicts=actor_info['solver']['max_conflicts'],
                     sat_restart_max=actor_info['solver']['sat_restart_max'],
                     lookahead_delta_fraction=actor_info['solver']['lookahead_delta_fraction'])
class Actor:
    def __init__(self, server, gpu_id, gpu_frac, actor_info):
        self.server     = server
        self.cfg        = server.get_config()
        self.cfg['dropout_training'] = False
        self.neuroquery = NeuroQuery(self.cfg, gpu_id, gpu_frac) if actor_info['tf'] else None

        self.cubers     = [mk_cuber(cuber_info, self.neuroquery) for cuber_info in actor_info['cubers']] if 'cubers' in actor_info else []
        self.branchers  = [mk_brancher(self.cfg, brancher_info, self.neuroquery) for brancher_info in actor_info['branchers']] if 'branchers' in actor_info else []

        self.actor_info  = actor_info

        self.sps = []
        for root, subdirs, files in os.walk(actor_info['dimacs_dir']):
            for dimacs in files:
                self.sps.append((dimacs, parse_dimacs(os.path.join(root, dimacs))))

    def pull_weights(self):
        if self.neuroquery is not None:
            self.neuroquery.set_weights(self.server.get_weights())

    def loop(self):
        while True:
            self.pull_weights()
            (dimacs, sp) = random.choice(self.sps)
            result = self.play_episode(dimacs, sp)
            if result is None: continue
            pre_datapoints, ps, cuber, brancher = result
            datapoints, esteps = self.build_datapoints(sp, pre_datapoints, ps)
            aer = ActorEpisodeResult(dimacs=dimacs, cuber=cuber, brancher=brancher, esteps=esteps, datapoints=datapoints)
            self.server.process_actor_episode(tuple(aer))

    def build_datapoints(self, sp, pre_datapoints, ps):
        datapoints = []
        esteps     = 1.0

        for pre_datapoint, p in reversed(list(zip(pre_datapoints, ps))):
            esteps = 1 + esteps / p
            datapoints.append(NeuroSATDatapoint(n_vars=sp.n_vars(),
                                                n_clauses=sp.n_clauses(),
                                                LC_idxs=pre_datapoint[0].LC_idxs,
                                                target_var=pre_datapoint[1],
                                                target_sl_esteps=esteps_to_sl_esteps(self.cfg, esteps)))

        return (datapoints if self.actor_info['train'] else []), esteps

    def play_episode(self, dimacs, sp):
        raise Exception("Abstract method")

class LookaheadActor(Actor):
    def play_episode(self, dimacs, sp):
        pre_datapoints = []
        ps = []

        s    = Z3Solver(sp=sp, opts=mk_zopts(self.actor_info))
        tfq  = s.to_tf_query(assumptions=[])
        assert(tfq.fvars)
        tfqr = self.neuroquery.query(sp.n_vars(), sp.n_clauses(), tfq.LC_idxs)

        while True:
            if tfq is None: break
            assert(tfq is not None)
            assert(tfqr is not None)

            status = s.check(assumptions=[])
            if status != Z3Status.unknown: break

            if self.actor_info['try_march_cu']:
                status, zlits = s.cube(assumptions=[], lookahead_reward="march_cu")
                if status != Z3Status.unknown: break

            fvar_logits           = tfqr['logits'][tfq.fvars]
            fvar_ps               = util.npsoftmax(fvar_logits)

            DIR_EPS, DIR_ASCALE   = self.actor_info['dirichlet']['epsilon'], self.actor_info['dirichlet']['ascale']
            fvar_ps               = DIR_EPS * np.random.dirichlet((DIR_ASCALE / np.size(fvar_ps)) * np.ones_like(fvar_ps)) + (1 - DIR_EPS) * fvar_ps

            n_lookahead           = min(self.actor_info['n_lookahead'], np.size(fvar_ps))

            # 'promising' free vars
            pfvars                = util.compute_top_k(fvar_ps, n_lookahead)

            if self.actor_info['consider_march_cu']:
                zvar_box    = np.array([zlits[0].var().idx()])
                pfvars      = np.union1d(pfvars, zvar_box)
                n_lookahead = np.size(pfvars)

            pfvar_tfqs   = [[None, None] for _ in range(n_lookahead)]
            pfvar_tfqrs  = [[None, None] for _ in range(n_lookahead)]
            pfvar_esteps = np.zeros(shape=(n_lookahead, 2))

            if self.actor_info['pull_every_step']:
                self.pull_weights()

            for pfvar_idx in range(n_lookahead):
                var = Var(tfq.fvars[pfvars[pfvar_idx]])
                for b_idx, b in enumerate([False, True]):
                    tfq_b = s.to_tf_query(assumptions=[Lit(var, b)])
                    if tfq_b.fvars:
                        pfvar_tfqs[pfvar_idx][b_idx]   = tfq_b
                        pfvar_tfqrs[pfvar_idx][b_idx]  = self.neuroquery.query(sp.n_vars(), sp.n_clauses(), tfq_b.LC_idxs)
                        pfvar_esteps[pfvar_idx][b_idx] = sl_esteps_to_esteps(self.cfg, pfvar_tfqrs[pfvar_idx][b_idx]['sl_esteps'])
                    else:
                        pfvar_esteps[pfvar_idx][b_idx] = 1.0

            # TODO(dselsam): there is probably a more principled way to do this
            pfvar_prior_ps         = util.npsoftmax(fvar_logits[pfvars] * self.actor_info['prior_tau'])
            pfvar_posterior_ps     = util.npsoftmax(- esteps_to_sl_esteps(self.cfg, np.sum(pfvar_esteps, axis=1)) * self.actor_info['posterior_tau'])

            pfvar_ps               = self.actor_info['prior_weight'] * pfvar_prior_ps + (1 - self.actor_info['prior_weight']) * pfvar_posterior_ps
            pfvar_ps               = pfvar_ps / np.sum(pfvar_ps) # not exactly 1 due to numerical issues

            best_pfvar_var         = np.random.choice(np.size(pfvar_ps), 1, p=pfvar_ps)[0]
            best_var               = tfq.fvars[pfvars[best_pfvar_var]]

            # importance sample
            is_ps     = pfvar_esteps[best_pfvar_var, :]
            is_ps     = is_ps / np.sum(is_ps)
            is_branch = np.random.choice(np.size(is_ps), 1, p=is_ps)[0]

            # we will compute the sl_esteps at the end, using the ps
            pre_datapoints.append((tfq, best_var))
            ps.append(is_ps[is_branch])

            # advance the solver and reuse the query results
            lit = Lit(Var(best_var), is_branch)
            s.add(lits=[lit])
            tfq  = pfvar_tfqs[best_pfvar_var][is_branch]
            tfqr = pfvar_tfqrs[best_pfvar_var][is_branch]

        return pre_datapoints, ps, "neuro-look-%d" % self.actor_info['n_lookahead'], "neuro-is"

class AsatActor(Actor):
    def play_episode(self, dimacs, sp):
        cuber    = random.choice(self.cubers)
        brancher = random.choice(self.branchers)

        s        = Z3Solver(sp=sp, opts=mk_zopts(self.actor_info))

        pre_datapoints = []
        ps             = []

        while True:
            status = s.check(assumptions=[])
            if status == Z3Status.sat:
                print("Warning: problem is SAT, not handling this yet")
                break
            elif status == Z3Status.unsat:
                break

            tfq = s.to_tf_query(assumptions=[])
            if not tfq.fvars:
                # TODO(dselsam, nikolaj): how can check return unknown if there are no free vars?
                break
            assert(tfq.fvars)

            var = cuber.cube(s, assumptions=[])
            if var is None:
                print("[%s] CUBER SOLVED PROBLEM" % cuber.name)
                break

            lit, plit = brancher.branch(s, var=var)

            pre_datapoints.append((tfq, var.idx()))
            ps.append(plit)
            s.add(lits=[lit])

        return pre_datapoints, ps, cuber.name, brancher.name


class RandomCuber:
    def __init__(self, name):
        self.name = name

    def cube(self, s, assumptions):
        # TODO(dselsam): could be faster
        tfq = s.to_tf_query(assumptions=assumptions)
        if len(tfq.fvars) == 0:
            return None
        else:
            fvars = list(tfq.fvars)
            return Var(random.choice(fvars))

class Z3Cuber:
    def __init__(self, name, lookahead_rewards):
        self.name              = name
        self.lookahead_rewards = lookahead_rewards

    def cube(self, s, assumptions):
        lr           = random.choice(self.lookahead_rewards)
        status, lits = s.cube(assumptions=assumptions, lookahead_reward=lr)
        if status != Z3Status.unknown: return None
        else:                          return lits[0].var()

class NeuroCuber:
    def __init__(self, name, tau, neuroquery):
        self.name       = name
        self.tau        = tau
        self.neuroquery = neuroquery

    def cube(self, s, assumptions):
        tfq = s.to_tf_query(assumptions=assumptions)
        if len(tfq.fvars) == 0:
            return None
        else:
            fvar_logits = self.neuroquery.query(s.sp().n_vars(), s.sp().n_clauses(), tfq.LC_idxs)['logits'][tfq.fvars]
            fvar_ps     = npsoftmax(fvar_logits * self.tau)
            fvar_choice = np.random.choice(np.size(fvar_ps), 1, p=fvar_ps)[0]
            return Var(tfq.fvars[fvar_choice])

## Branchers

class RandomBrancher:
    def __init__(self, cfg, name):
        self.cfg  = cfg
        self.name = name

    def branch(self, s, var):
        return Lit(var, random.random() < 0.5), 0.5

class NeuroISBrancher:
    def __init__(self, cfg, name, neuroquery):
        self.cfg        = cfg
        self.name       = name
        self.neuroquery = neuroquery

    def branch(self, s, var):
        tfqs   = [s.to_tf_query(assumptions=[Lit(var, b)]) for b in [False, True]]
        esteps = np.zeros(2)

        for i in range(2):
            if not tfqs[i].fvars:
                esteps[i] = 1.0
            else:
                sl_esteps = self.neuroquery.query(s.sp().n_vars(), s.sp().n_clauses(), tfqs[i].LC_idxs)['sl_esteps']
                esteps[i] = sl_esteps_to_esteps(self.cfg, sl_esteps)

        is_ps     = esteps / np.sum(esteps)
        is_branch = np.random.choice(np.size(is_ps), 1, p=is_ps)[0]

        return Lit(var, is_branch), is_ps[is_branch]

## Makers

def mk_cuber(cuber_info, neuroquery):
    if cuber_info['kind'] == "random":
        return RandomCuber(cuber_info['name'])
    elif cuber_info['kind'] == "z3":
        return Z3Cuber(cuber_info['name'], cuber_info['lookahead_rewards'])
    elif cuber_info['kind'] == "neuro":
        return NeuroCuber(cuber_info['name'], cuber_info['tau'], neuroquery)
    else:
        raise Exception("unexpected cuber kind '%s'" % cuber_info['kind'])

def mk_brancher(cfg, brancher_info, neuroquery):
    if brancher_info['kind'] == "random":
        return RandomBrancher(cfg, brancher_info['name'])
    elif brancher_info['kind'] == "neuro-is":
        return NeuroISBrancher(cfg, brancher_info['name'], neuroquery)
    else:
        raise Exception("unexpected brancher kind '%s'" % brancher_info['kind'])

def mk_actor(server, gpu_id, gpu_frac, actor_info):
    if actor_info['kind'] == 'lookahead':
        return LookaheadActor(server, gpu_id, gpu_frac, actor_info)
    elif actor_info['kind'] == 'asat':
        return AsatActor(server, gpu_id, gpu_frac, actor_info)
    else:
        raise Exception("only lookahead actors currently supported")
