/*
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
*/
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
