#include "decl.hpp"
#include <choosi_core/optimization/qp_barrier_cqn.hpp>

namespace cs = choosi_core;

void qp_barrier_cqn(py::module_& m)
{
    using solver_t = cs::optimization::QPBarrierCQN<double>;
    using value_t = typename solver_t::value_t;
    using vec_value_t = typename solver_t::vec_value_t;
    using rowmat_value_t = typename solver_t::rowmat_value_t;
    py::class_<solver_t>(m, "QPBarrierCQN", "QP Barrier CQN solver.")
        .def(py::init<
            const Eigen::Ref<const rowmat_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            size_t,
            Eigen::Ref<rowmat_value_t>,
            size_t,
            value_t,
            bool,
            size_t,
            value_t,
            value_t 
        >(),
            py::arg("quad").noconvert(),
            py::arg("linear").noconvert(),
            py::arg("signs").noconvert(),
            py::arg("lmda"),
            py::arg("n_threads"),
            py::arg("quad_chol").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("armijo"),
            py::arg("armijo_max_iters"),
            py::arg("armijo_c"),
            py::arg("armijo_tau")
        )
        .def("solve", &solver_t::solve)
        ;
}

void register_optimization(py::module_& m)
{
    qp_barrier_cqn(m);
}