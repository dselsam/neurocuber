/**
   This module provides generic utilities for dealing with SAT problems.
   It is "excessively-typed" to prevent subtle errors.
**/
#pragma once
#include <pybind11/pybind11.h>
#include <vector>

using std::string;
using std::vector;
using std::pair;

struct Var {
  unsigned _idx;
  Var(unsigned idx);
  unsigned idx() const;

  string repr() const;
  bool operator==(const Var & other) const;
  unsigned hash() const;
};

namespace std {
  template <> struct hash<Var> { std::size_t operator()(const Var& var) const { return var.hash(); }};
}

struct Lit {
  Var      _var;
  bool     _sign;
  Lit(Var var, bool sign);

  Var var() const;
  bool sign() const;

  unsigned vidx(unsigned n_vars) const;
  int ilit() const;
  Lit flip() const;

  string repr() const;
  bool operator==(const Lit & other) const;
  unsigned hash() const;
};

namespace std {
  template <> struct hash<Lit> { std::size_t operator()(const Lit& lit) const { return lit.hash(); }};
}

typedef vector<Lit> Clause;

struct SATProblem {
  unsigned       _n_vars;
  vector<Clause> _clauses;
  SATProblem(unsigned n_vars, vector<Clause> const & clauses);
  unsigned n_vars() const;
  unsigned n_lits() const;
  unsigned n_clauses() const;
  unsigned n_cells() const;
  vector<Clause> const & clauses() const;
  void print() const;
};

Lit ilit_to_lit(int ilit);
SATProblem parse_dimacs(string const & filename);

namespace py = pybind11;

void init_py_problem_module(py::module & m);
