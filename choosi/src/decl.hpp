#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
// #include <snpkit_core/util/types.hpp>

namespace py = pybind11;

void register_io(py::module_&);