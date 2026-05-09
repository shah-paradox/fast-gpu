#include "solver.h"
#include <stdlib.h>

PYBIND11_MODULE(core_openmp, m) {
  m.def("likwid_init", []() {
  });
  m.def("likwid_close", []() {
  });

  py::class_<OpenMPEquSolver>(m, "EquSolver")
      .def(py::init<int>())
      .def("partition", &OpenMPEquSolver::partition)
      .def("reset", &OpenMPEquSolver::reset)
      .def("sync", &OpenMPEquSolver::sync)
      .def("step", &OpenMPEquSolver::step);
  py::class_<OpenMPGridSolver>(m, "GridSolver")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPGridSolver::reset)
      .def("sync", &OpenMPGridSolver::sync)
      .def("step", &OpenMPGridSolver::step);
  py::class_<OpenMPMultigridSolver>(m, "MultigridSolver")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPMultigridSolver::reset)
      .def("sync", &OpenMPMultigridSolver::sync)
      .def("step", &OpenMPMultigridSolver::step);
  py::class_<OpenMPSolverV3>(m, "SolverV3")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPSolverV3::reset)
      .def("sync", &OpenMPSolverV3::sync)
      .def("step", &OpenMPSolverV3::step);
  py::class_<OpenMPSolverV4>(m, "SolverV4")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPSolverV4::reset)
      .def("sync", &OpenMPSolverV4::sync)
      .def("step", &OpenMPSolverV4::step);
  py::class_<OpenMPSolverV5>(m, "SolverV5")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPSolverV5::reset)
      .def("sync", &OpenMPSolverV5::sync)
      .def("step", &OpenMPSolverV5::step);
}
