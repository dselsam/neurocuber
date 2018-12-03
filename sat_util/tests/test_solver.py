from sat_util import *
from nose.tools import assert_equals, assert_not_equal, assert_true, assert_raises
import numpy as np
import os

TEST_DIR = "/home/dselsam/alphacuber/tests"

def opts(max_conflicts):
    return Z3Options(max_conflicts=max_conflicts, sat_restart_max=0)

def test_Z3Solver_basics():
    sp = parse_dimacs(os.path.join(TEST_DIR, "test1.dimacs"))

    s = Z3Solver(sp, opts(max_conflicts=0))
    assert_equals(s.check([]), Z3Status.unknown)

    s = Z3Solver(sp, opts(max_conflicts=10))
    assert_equals(s.check([]), Z3Status.sat)

def check_to_tf_query(dimacs, expected):
    sp = parse_dimacs(dimacs)
    s = Z3Solver(sp, opts(max_conflicts=0))
    assert_equals(s.check([]), Z3Status.unknown)

    tfq = s.to_tf_query(expected['trail'])

    assert_equals(tfq.fvars, expected['fvars'])
    assert_true((tfq.LC_idxs == expected['LC_idxs']).all())

def test_to_tf_query():
    TESTS = {
        os.path.join(TEST_DIR, "test1.dimacs") : {
            'trail':[],
            'fvars':[0, 1, 2, 3],
            'LC_idxs': np.array([
                [0, 0], [1, 0], [2, 0], [3, 0],
                [6, 1], [7, 1], [8, 1],
                [0, 3], [2, 3],
                [9, 4], [7, 4], [6, 4],
                [3, 6], [7, 6]
            ])
        },

        os.path.join(TEST_DIR, "test1.dimacs") : {
            'trail':[Lit(Var(3), False)],
            'fvars':[0, 1, 2],
            'LC_idxs': np.array([
                [6, 1], [7, 1], [8, 1],
                [0, 3], [2, 3],
                [7, 4], [6, 4],
            ])
        }
    }

    for dimacs, expected in TESTS.items():
        yield check_to_tf_query, dimacs, expected

def test_unsat_core():
    sp = parse_dimacs(os.path.join(TEST_DIR, "test1.dimacs"))
    s = Z3Solver(sp, opts(max_conflicts=0))
    assert_equals(s.check([Lit(Var(0), False), Lit(Var(4), True), Lit(Var(1), False)]), Z3Status.unsat)
    core = s.unsat_core()
    # TODO(dselsam): support < on lits
    assert_equals(len(core), 2)
    assert_true(Lit(Var(0), False) in core)
    assert_true(Lit(Var(1), False) in core)