set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -mavx2 -fopenmp -DLIKWID_PERFMON")

pybind11_add_module(core_openmp solver.cc equ.cc grid.cc multigrid.cc v3.cc v4.cc v5.cc)

find_library(LIKWID_LIB likwid)
target_link_libraries(core_openmp PRIVATE ${LIKWID_LIB})
