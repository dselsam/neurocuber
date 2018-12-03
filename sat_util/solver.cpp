#include "except.h"
#include "solver.h"
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include <sstream>

using std::unordered_map;
using std::unordered_set;

// Util
static Z3Status check_result_to_Z3Status(z3::check_result const & cr) {
  switch (cr) {
  case z3::unknown: return Z3Status::UNKNOWN;
  case z3::unsat: return Z3Status::UNSAT;
  case z3::sat: return Z3Status::SAT;
  }
  throw SatUtilException("check_result_to_Z3Status: unexpected status from z3");
}

static Z3Status zpropagate(Z3Options const & opts, z3::solver & s) {
  s.set(":max_conflicts", (unsigned) 0);
  z3::check_result cr = s.check();
  s.set(":max_conflicts", opts.max_conflicts);
  return check_result_to_Z3Status(cr);
}

static z3::solver ztranslate(z3::solver & s0) {
  return z3::solver(s0.ctx(), s0, z3::solver::translate());
}

// Z3Options
Z3Options::Z3Options(unsigned max_conflicts, unsigned sat_restart_max):
  max_conflicts(max_conflicts), sat_restart_max(sat_restart_max) {}

// Z3Solver

z3::expr Z3Solver::var_to_zvar(Var const & var) const {
  if (var.idx() >= _var_to_zvar.size()) {
    throw SatUtilException("var_to_zvar on invalid var");
  }
  return _var_to_zvar[var.idx()];
}

z3::expr Z3Solver::lit_to_zlit(Lit const & lit) const {
  z3::expr zvar = var_to_zvar(lit.var());
  return lit.sign() ? !zvar : zvar;
}

Var Z3Solver::zvar_to_var(z3::expr const & zvar) const {
  if (!zvar.is_const() || !_zvar_to_var.count(zvar)) {
    throw SatUtilException("zvar_to_var on invalid zvar");
  }
  return _zvar_to_var.at(zvar);
}

Lit Z3Solver::zlit_to_lit(z3::expr const & zlit) const {
  bool sign = zlit.is_not();
  z3::expr zvar = sign ? zlit.arg(0) : zlit;
  Var var = zvar_to_var(zvar);
  return Lit(var, sign);
}

z3::solver Z3Solver::zclone_and_set(vector<Lit> const & assumptions) {
  z3::solver s = ztranslate(_zsolver);
  set_zsolver_params(s);
  for (Lit const & lit : assumptions) {
    s.add(lit_to_zlit(lit));
  }
  return s;
}

TFQuery Z3Solver::to_tf_query(vector<Lit> const & assumptions) {
  z3::solver s = zclone_and_set(assumptions);
  // we propagate here just in case
  // note: result might not have free vars
  zpropagate(_opts, s);

  // we don't know the number of cells yet
  vector<pair<unsigned, unsigned> > idxs;
  unsigned n_cells = 0;

  // collect units
  bool units[_sp.n_lits()] = { false };

  z3::expr_vector zunits = s.units();
  for (unsigned u_idx = 0; u_idx < zunits.size(); ++u_idx) {
    units[zlit_to_lit(zunits[u_idx]).vidx(_sp.n_vars())] = true;
  }

  // cmask
  for (unsigned c_idx = 0; c_idx < _sp.clauses().size(); ++c_idx) {
    Clause clause;
    bool has_unit = false;
    for (Lit const & lit : _sp.clauses()[c_idx]) {
      if (units[lit.vidx(_sp.n_vars())]) {
	has_unit = true;
	break;
      } else if (!units[lit.flip().vidx(_sp.n_vars())]) {
	clause.push_back(lit);
      }
    }

    if (clause.size() > 1 && !has_unit) {
      for (Lit const & lit : clause) {
	idxs.push_back({lit.vidx(_sp.n_vars()), c_idx});
      }
      n_cells += clause.size();
    }
  }

  TFQuery tfq;
  tfq.LC_idxs = Eigen::MatrixXi(n_cells, 2);
  for (unsigned cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
    tfq.LC_idxs(cell_idx, 0) = idxs[cell_idx].first;
    tfq.LC_idxs(cell_idx, 1) = idxs[cell_idx].second;
  }

  z3::expr_vector zfvars = s.non_units();
  for (unsigned fv_idx = 0; fv_idx < zfvars.size(); ++fv_idx) {
    tfq.fvars.push_back(zlit_to_lit(zfvars[fv_idx]).var().idx());
  }
  std::sort(tfq.fvars.begin(), tfq.fvars.end());
  return tfq;
}

void Z3Solver::set_zsolver_params(z3::solver & s) {
  s.set(":lookahead.cube.cutoff", "depth");
  s.set(":lookahead.cube.depth", (unsigned)1);
  s.set(":unsat_core", true);
  s.set(":core.minimize", true);
  s.set(":core.minimize_partial", true);
  // s.set(":sat.force_cleanup", true);

  s.set(":max_conflicts", _opts.max_conflicts);
  s.set(":sat.restart.max", _opts.sat_restart_max);
}

void Z3Solver::print() const {
  std::cout << "---Z3Solver---" << std::endl;
  std::cout << "n_units: " << _zsolver.units().size() << std::endl;
  std::cout << "n_non_units: " << _zsolver.non_units().size() << std::endl;
  z3::expr_vector clauses = _zsolver.assertions();
  for (unsigned i = 0; i < clauses.size(); ++i) {
    std::cout << "[" << i << "] ";
    z3::expr e = clauses[i];
    for (unsigned j = 0; j < e.num_args(); ++j) {
      std::cout << e.arg(j) << " ";
    }
    std::cout << std::endl;
  }
}

void Z3Solver::validate_cube(z3::expr_vector const & cube) {
  if (cube.size() == 0) {
    throw SatUtilException("z3::cube() returned empty cube");
  } else if (cube.size() > 1) {
    throw SatUtilException("unexpected cube of size > 1");
  }
}

void Z3Solver::add(vector<Lit> const & lits) {
  for (Lit const & lit : lits) {
    _zsolver.add(lit_to_zlit(lit));
  }
}

pair<Z3Status, vector<Lit>> Z3Solver::cube(vector<Lit> const & assumptions, string const & lookahead_reward) {
  z3::solver s = zclone_and_set(assumptions);
  s.set(":lookahead.reward", lookahead_reward.c_str());

  z3::solver::cube_generator cg    = s.cubes();
  z3::solver::cube_iterator  start = cg.begin();
  z3::solver::cube_iterator  end   = cg.end();

  if (start == end) { return { Z3Status::UNSAT, {} }; }

  z3::expr_vector cube1 = *start;
  validate_cube(cube1);

  assert(cube1.size() == 1);
  if (cube1[0].is_true()) { return { Z3Status::SAT, {} }; }

  ++start;
  if (start == end) { /* failed lit */ return { Z3Status::UNKNOWN, { zlit_to_lit(cube1[0]) } }; }

  z3::expr_vector cube2 = *start;
  validate_cube(cube2);

  if (cube1 == cube2) {
    throw SatUtilException("Both cubes are the same: did you forget to increment the iterator?");
  }

  ++start;
  if (start != end) {
    throw SatUtilException("z3::cube() returned more than two cubes");
  }
  return { Z3Status::UNKNOWN, { zlit_to_lit(cube1[0]), zlit_to_lit(cube2[0]) } };
}

vector<Lit> Z3Solver::unsat_core() {
  vector<Lit> core;
  z3::expr_vector zcore = _zsolver.unsat_core();
  for (unsigned i = 0; i < zcore.size(); ++i) {
    core.push_back(zlit_to_lit(zcore[i]));
  }
  return core;
}

Z3Status Z3Solver::check(vector<Lit> const & lits) {
  vector<z3::expr> assumptions;
  for (Lit const & lit : lits) {
    assumptions.push_back(lit_to_zlit(lit));
  }
  return check_result_to_Z3Status(_zsolver.check(assumptions.size(), assumptions.data()));
}

SATProblem const & Z3Solver::sp() const {
  return _sp;
}

Z3Solver::Z3Solver(SATProblem const & sp, Z3Options const & opts):
  _sp(sp), _opts(opts), _zctx(), _zsolver(_zctx, "QF_FD")  {
  set_zsolver_params(_zsolver);

  // create zvars
  for (unsigned v_idx = 0; v_idx < _sp.n_vars(); ++v_idx) {
    std::string name = "x_" + std::to_string(v_idx);
    z3::expr zvar = _zsolver.ctx().bool_const(name.c_str());
    _var_to_zvar.push_back(zvar);
    _zvar_to_var.insert({zvar, Var(v_idx)});
  }

  // add clauses to solver
  for (Clause const & clause : _sp.clauses()) {
    z3::expr_vector args(_zctx);
    for (Lit const & lit : clause) {
      args.push_back(lit_to_zlit(lit));
    }
    _zsolver.add(z3::mk_or(args));
  }
}

// Pybind11
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

void init_py_solver_module(py::module & m) {
  py::class_<Z3Options>(m, "Z3Options")
    .def(py::init<unsigned, unsigned>(), py::arg("max_conflicts"), py::arg("sat_restart_max"));

  py::enum_<Z3Status>(m, "Z3Status")
    .value("unknown", Z3Status::UNKNOWN)
    .value("unsat", Z3Status::UNSAT)
    .value("sat", Z3Status::SAT);

  py::class_<TFQuery>(m, "TFQuery")
    .def_readonly("fvars", &TFQuery::fvars)
    .def_readonly("LC_idxs", &TFQuery::LC_idxs);

  py::class_<Z3Solver>(m, "Z3Solver")
    .def(py::init<SATProblem const &, Z3Options const &>(), py::arg("sp"), py::arg("opts"))
    .def("sp", &Z3Solver::sp)
    .def("add", &Z3Solver::add, py::arg("lits"))
    .def("check", &Z3Solver::check, py::arg("assumptions"), py::call_guard<py::gil_scoped_release>())
    .def("unsat_core", &Z3Solver::unsat_core)
    .def("to_tf_query", &Z3Solver::to_tf_query, py::arg("assumptions"), py::call_guard<py::gil_scoped_release>())
    .def("print", &Z3Solver::print)
    .def("cube", &Z3Solver::cube, py::arg("assumptions"), py::arg("lookahead_reward"), py::call_guard<py::gil_scoped_release>());
}
