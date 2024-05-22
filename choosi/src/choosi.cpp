#include "decl.hpp"

PYBIND11_MODULE(choosi_core, m) {
    auto m_matrix = m.def_submodule("matrix", "Matrix submodule.");
    register_matrix(m_matrix);
}