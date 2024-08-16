#include "decl.hpp"

PYBIND11_MODULE(choosi_core, m) {
    auto m_matrix = m.def_submodule("matrix", "Matrix submodule.");
    register_matrix(m_matrix);
    auto m_distr = m.def_submodule("distr", "Distr submodule.");
    register_distr(m_distr);
}