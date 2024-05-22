#include "decl.hpp"
#include <snpkit_core/core.hpp>
#include <snpkit_core/util/types.hpp>

namespace cs = choosi_core;

PYBIND11_MODULE(snpkit_core, m) {
    auto m_io = m.def_submodule("io", "IO submodule.");
    register_io(m_io);

    m.def("to_sample_major_", &cs::to_sample_major<int8_t>); 
}