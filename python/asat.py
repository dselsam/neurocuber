import numpy as np
import math
import random
from util import npsoftmax
from sat_util import *

def play_asat(s, cuber, brancher):
    trail = []

    while True:
        status = s.check(assumptions=trail)
        if status == Z3Status.sat:
            print("Warning: problem is SAT, not handling this yet")
            return None
        elif status == Z3Status.unsat:
            core = s.unsat_core()
            assert(len(core) <= len(trail))
            return trail, core

        var = cuber.cube(s, assumptions=trail)
        if var is None:
            print("[%s|%s] FAILED(%s)" % (cuber.name, brancher.name, len(trail)))
            return None
        lit_choices = [Lit(var, False), Lit(var, True)]
        lit = brancher.branch(s, assumptions=trail, lit_choices=lit_choices)
        trail.append(lit)

# Cubers

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
            ok_logits = self.neuroquery.query(s.sp().n_vars(), s.sp().n_clauses(), tfq.LC_idxs)['logits'][tfq.fvars]
            ok_ps     = npsoftmax(ok_logits * self.tau)
            ok_choice = np.random.choice(np.size(ok_ps), 1, p=ok_ps)[0]
            return Var(tfq.fvars[ok_choice])

## Branchers

class RandomBrancher:
    def __init__(self, name):
        self.name = name

    def branch(self, s, assumptions, lit_choices):
        return random.choice(lit_choices)

class NeuroBrancher:
    def __init__(self, name, tau, neuroquery):
        self.name       = name
        self.tau        = tau
        self.neuroquery = neuroquery

    def branch(self, s, assumptions, lit_choices):
        tfqs = [s.to_tf_query(assumptions=assumptions + [lit]) for lit in lit_choices]
        if len(tfqs[0].fvars) == 0:
            # TODO(dselsam): look-ahead here
            return lit_choices[1]
        elif len(tfqs[1].fvars) == 0:
            # TODO(dselsam): look-ahead here
            return lit_choices[0]
        else:
            adversary_vs = -np.array([self.neuroquery.query(s.sp().n_vars(), s.sp().n_clauses(), tfq.LC_idxs)['v'] for tfq in tfqs])
            assert(np.size(adversary_vs) == 2)
            adversary_ps     = npsoftmax(adversary_vs * self.tau)
            adversary_choice = np.random.choice(2, 1, p=adversary_ps)[0]
        assert(adversary_choice in [0, 1])
        return lit_choices[adversary_choice]

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

def mk_brancher(brancher_info, neuroquery):
    if brancher_info['kind'] == "random":
        return RandomBrancher(brancher_info['name'])
    elif brancher_info['kind'] == "neuro":
        return NeuroBrancher(brancher_info['name'], brancher_info['tau'], neuroquery)
    else:
        raise Exception("unexpected brancher kind '%s'" % brancher_info['kind'])
