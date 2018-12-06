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
#include <Eigen/Dense>
#include "z3++.h"
#include "problem.h"
#include <unordered_map>

struct z3_expr_hash { unsigned operator()(z3::expr const & e) const { return e.hash(); } };
struct z3_expr_eq { bool operator()(z3::expr const & e1, z3::expr const & e2) const { return z3::eq(e1, e2); } };

template<typename T>
using z3_expr_map = typename std::unordered_map<z3::expr, T, z3_expr_hash, z3_expr_eq>;

enum class Z3Status { UNKNOWN, UNSAT, SAT };

struct Z3Options {
  unsigned max_conflicts;
  unsigned sat_restart_max;

  Z3Options(unsigned max_conflicts, unsigned sat_restart_max);
};

struct TFQuery {
  vector<unsigned> fvars;
  Eigen::MatrixXi LC_idxs;
};

class Z3Solver {
 private:
  SATProblem        _sp;
  Z3Options         _opts;
  z3::context       _zctx;
  z3::solver        _zsolver;

  vector<z3::expr>  _var_to_zvar;
  z3_expr_map<Var>  _zvar_to_var;

  z3::expr var_to_zvar(Var const & var) const;
  z3::expr lit_to_zlit(Lit const & lit) const;
  Var zvar_to_var(z3::expr const & zvar) const;
  Lit zlit_to_lit(z3::expr const & zlit) const;

  void set_params();
  void validate_cube(z3::expr_vector const & cube);

 public:
  Z3Solver(SATProblem const & sp, Z3Options const & opts);

  void push();
  void pop();
  void reset();

  Z3Status propagate();
  SATProblem const & sp() const;
  pair<Z3Status, vector<Lit>> cube(string const & lookahead_reward, float lookahead_delta_fraction);
  void add(vector<Lit> const & lits);
  Z3Status check();
  pair<Z3Status, vector<Lit>> check_core(vector<Lit> const & assumptions);
  TFQuery to_tf_query();

  void print() const;
};

#include <pybind11/pybind11.h>
void init_py_solver_module(py::module & m);
