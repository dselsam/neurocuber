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
from sat_util import *
from nose.tools import assert_equals, assert_not_equal, assert_true, assert_raises
import numpy as np
import os

TEST_DIR = "/home/dselsam/alphacuber/tests"

def opts(max_conflicts):
    return Z3Options(max_conflicts=max_conflicts, sat_restart_max=0, lookahead_delta_fraction=0.1)

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

def test_scopes():
    sp = parse_dimacs(os.path.join(TEST_DIR, "test1.dimacs"))
    s = Z3Solver(sp, opts(max_conflicts=0))
    assert_equals(s.check([]), Z3Status.unknown)
    s.push()
    s.add([Lit(Var(0), False), Lit(Var(4), True), Lit(Var(1), False)])
    assert_equals(s.check([]), Z3Status.unsat)
    assert_equals(s.check([]), Z3Status.unsat)
    s.pop()
    assert_equals(s.check([]), Z3Status.unknown)
