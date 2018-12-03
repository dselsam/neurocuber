from sat_util import *
from nose.tools import assert_equals, assert_not_equal
import pickle
import gzip

def test_var_idx():
    v = Var(3)
    assert_equals(3, v.idx())

def test_var_eq():
    v1 = Var(3)
    v2 = Var(3)
    assert_equals(v1, v2)

def test_var_neq():
    v1 = Var(3)
    v2 = Var(4)
    assert_not_equal(v1, v2)

def test_var_hash_eq():
    v1 = Var(3)
    v2 = Var(3)
    assert_equals(hash(v1), hash(v2))

def test_var_hash_neq():
    v1 = Var(3)
    v2 = Var(4)
    assert_not_equal(hash(v1), hash(v2))

def test_var_pickle():
    v1a = Var(3)
    v1b = Var(7)
    with gzip.GzipFile('.tmp', 'w') as f:
        pickle.dump((v1a, v1b), f)

    with gzip.GzipFile('.tmp', 'r') as f:
        v2a, v2b = pickle.load(f)
    assert_equals(v1a, v2a)
    assert_equals(v1b, v2b)

def test_lit_var_sign():
    v = Var(3)
    b = False
    l = Lit(v, b)
    assert_equals(l.var(), v)
    assert_equals(l.sign(), b)

def test_lit_flip():
    v = Var(3)
    b = False
    l = Lit(v, b).flip()
    assert_equals(l.var(), v)
    assert_equals(l.sign(), not b)

def test_lit_vidx():
    l = Lit(Var(3), False)
    assert_equals(l.vidx(100), 3)
    assert_equals(l.flip().vidx(100), 103)

def test_lit_ilit():
    l = Lit(Var(3), False)
    assert_equals(l.ilit(), 4)
    assert_equals(l.flip().ilit(), -4)

def test_lit_pickle():
    l1a = Lit(Var(3), False)
    l1b = Lit(Var(5), True)
    with gzip.GzipFile('.tmp', 'w') as f:
        pickle.dump((l1a, l1b), f)

    with gzip.GzipFile('.tmp', 'r') as f:
        (l2a, l2b) = pickle.load(f)
    assert_equals(l1a, l2a)
    assert_equals(l1b, l2b)

def test_sat_problem():
    n_vars = 3
    clauses = [[Lit(Var(0), False), Lit(Var(1), True)], [Lit(Var(2), True), Lit(Var(1), False)]]
    sp = SATProblem(n_vars=n_vars, clauses=clauses)
    assert_equals(sp.n_vars(), n_vars)
    assert_equals(sp.n_clauses(), len(clauses))
    assert_equals(sp.clauses(), clauses)
    assert_equals(sp.n_cells(), sum([len(c) for c in clauses]))

def test_parse_dimacs():
    sp = parse_dimacs("/home/dselsam/alphacuber/tests/test1.dimacs")
    assert_equals(sp.n_vars(), 6)
    assert_equals(len(sp.clauses()), 7)
    assert_equals(sp.clauses()[0], [Lit(Var(i), False) for i in range(4)])
