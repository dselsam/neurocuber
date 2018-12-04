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
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include "problem.h"

using std::vector;
using std::unordered_map;
using std::string;
using std::pair;

// Var
Var::Var(unsigned idx): _idx(idx) {}
unsigned Var::idx() const { return _idx; }
string Var::repr() const { return std::string("Var(") + std::to_string(idx()) + ")"; }
bool Var::operator==(const Var & other) const { return idx() == other.idx(); }
unsigned Var::hash() const { return idx(); }

// Lit
Lit::Lit(Var var, bool sign): _var(var), _sign(sign) {}
Var Lit::var() const { return _var; }
bool Lit::sign() const { return _sign; }
unsigned Lit::vidx(unsigned n_vars) const { return sign() ? var().idx() + n_vars : var().idx(); }
int Lit::ilit() const { return sign() ? -(var().idx() + 1) : (var().idx() + 1); }
Lit Lit::flip() const { return Lit(var(), !sign()); }
string Lit::repr() const { return std::string("Lit(") + var().repr() + ", " + (sign() ? "1" : "0") + ")"; }
bool Lit::operator==(const Lit & other) const { return var() == other.var() && sign() == other.sign(); }
unsigned Lit::hash() const { return var().hash() + (sign() ? 48371 : 0); }

// SAT Problem
SATProblem::SATProblem(unsigned n_vars, vector<Clause> const & clauses):
  _n_vars(n_vars), _clauses(clauses) {}

unsigned SATProblem::n_vars() const { return _n_vars; }
unsigned SATProblem::n_lits() const { return 2 * n_vars(); }
unsigned SATProblem::n_clauses() const { return clauses().size(); }
unsigned SATProblem::n_cells() const {
  unsigned n_cells = 0;
  for (Clause const & clause : clauses()) {
    n_cells += clause.size();
  }
  return n_cells;
}
vector<Clause> const & SATProblem::clauses() const { return _clauses; }

void SATProblem::print() const {
  std::cout << "---SATProblem---" << std::endl;
  std::cout << "n_vars: " << n_vars() << std::endl;
  std::cout << "n_clauses: " << clauses().size() << std::endl;
  for (unsigned i = 0; i < clauses().size(); ++i) {
    std::cout << "[" << i << "] ";
    for (Lit const & lit : clauses()[i]) {
      std::cout << lit.ilit() << " ";
    }
    std::cout << std::endl;
  }
}

// Util
Lit ilit_to_lit(int ilit) { return Lit(abs(ilit) - 1, ilit < 0); }

SATProblem parse_dimacs(std::string const & filename) {
  std::ifstream file(filename);
  string line;
  bool seen_p = false;

  unsigned n_vars, n_clauses;
  vector<Clause> clauses;

  while (getline(file, line)) {
    std::istringstream iss(line);
    string result;
    if (getline(iss, result, ' ')) {
      if (result == "c") {
        // comment
        continue;
      } else if (result == "p") {
        // header
	if (seen_p) {
	  throw std::runtime_error("Error parsing dimacs: multiple header lines");
	}
        seen_p = true;

        string token;
        getline(iss, token, ' ');
        assert(token == "cnf");

        getline(iss, token, ' ');
        n_vars = stoi(token);

        getline(iss, token);
        n_clauses = stoi(token);
      } else {
        // clause
        assert(seen_p);
	vector<Lit> lits;
        string token = result;
        do {
          if (token == "0") {
            break;
          } else {
	    lits.push_back(ilit_to_lit(stoi(token)));
          }
        } while (getline(iss, token, ' '));
	clauses.push_back(lits);
      }
    }
  }
  if (!seen_p) { throw std::runtime_error("Error parsing dimacs: no header"); }
  if (n_clauses != clauses.size()) { throw std::runtime_error("Error parsing dimacs: number of clauses do not match"); }
  return SATProblem(n_vars, clauses);
}

// Pybind11

#include <pybind11/stl.h>

void init_py_problem_module(py::module & m) {
  py::class_<Var>(m, "Var")
    .def(py::init<unsigned>(), py::arg("idx"))
    .def("idx", &Var::idx)
    .def("__str__", &Var::repr)
    .def("__repr__", &Var::repr)
    .def("__eq__", &Var::operator==)
    .def("__hash__", &Var::hash)
    .def(py::pickle([](const Var & var) { return var.idx(); },
		    [](unsigned idx) { return Var(idx); }));

  py::class_<Lit>(m, "Lit")
    .def(py::init<Var, bool>(), py::arg("var"), py::arg("sign"))
    .def("var", &Lit::var)
    .def("sign", &Lit::sign)
    .def("vidx", &Lit::vidx)
    .def("ilit", &Lit::ilit)
    .def("flip", &Lit::flip)
    .def("__str__", &Lit::repr)
    .def("__repr__", &Lit::repr)
    .def("__eq__", &Lit::operator==)
    .def("__hash__", &Lit::hash)
    .def(py::pickle([](const Lit & lit) { return py::make_tuple(lit.var().idx(), lit.sign()); },
		    [](py::tuple t) { return Lit(Var(t[0].cast<unsigned>()), t[1].cast<bool>()); }));

  py::class_<SATProblem>(m, "SATProblem")
    .def(py::init<unsigned, vector<Clause> const &>(), py::arg("n_vars"), py::arg("clauses"))
    .def("n_vars", &SATProblem::n_vars)
    .def("n_lits", &SATProblem::n_lits)
    .def("n_clauses", &SATProblem::n_clauses)
    .def("n_cells", &SATProblem::n_cells)
    .def("clauses", &SATProblem::clauses, py::return_value_policy::reference_internal)
    .def("print", &SATProblem::print);

  m.def("parse_dimacs", &parse_dimacs, py::return_value_policy::copy);
}
