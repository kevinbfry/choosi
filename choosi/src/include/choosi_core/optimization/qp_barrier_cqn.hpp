#pragma once
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <adelie_core/matrix/utils.hpp>

namespace choosi_core {
namespace optimization {

template <class ValueType>
class QPBarrierBase
{
public:
    using value_t = ValueType;
    using vec_value_t = Eigen::Array<value_t, 1, Eigen::Dynamic>;

    const size_t max_iters;
    const value_t tol;
    const bool armijo;
    const size_t armijo_max_iters;
    const value_t armijo_c;
    const value_t armijo_tau;

    explicit QPBarrierBase(
        size_t max_iters,
        value_t tol,
        bool armijo,
        size_t armijo_max_iters,
        value_t armijo_c,
        value_t armijo_tau
    ):
        max_iters(max_iters),
        tol(tol),
        armijo(armijo),
        armijo_max_iters(armijo_max_iters),
        armijo_c(armijo_c),
        armijo_tau(armijo_tau)
    {
        if (tol <= 0) {
            throw std::runtime_error("tol must be > 0.");
        }
        if (armijo_c <= 0) {
            throw std::runtime_error("armijo_c must be > 0.");
        }
        if (armijo_tau <= 0 || armijo_tau >= 1) {
            throw std::runtime_error("armijo_tau must be in (0,1).");
        }
    }

    template <class GradFType, class InvHessGradFType, class ObjectiveFType>
    void solve(
        vec_value_t& z,
        GradFType grad_f,
        InvHessGradFType inv_hess_grad_f,
        ObjectiveFType objective_f
    )
    {
        const auto d = z.size();

        vec_value_t z_prev(d);
        vec_value_t grad(d);
        vec_value_t grad_prev(d);
        vec_value_t inv_hess_grad(d);

        grad_f(z, grad);

        for (int i = 0; i < max_iters; ++i) {
            value_t step_size = 1;

            // save new previous quantities
            z_prev.swap(z);
            grad_prev.swap(grad);

            // Newton update
            inv_hess_grad_f(z_prev, grad_prev, inv_hess_grad);
            z = z_prev - step_size * inv_hess_grad;

            // armijo line search
            if (armijo) {
                value_t t = armijo_c * std::abs((grad_prev * inv_hess_grad).sum());
                value_t obj_prev = objective_f(z_prev);
                value_t obj = objective_f(z);
                int ct = 0;
                for (; ct < armijo_max_iters; ++ct) {
                    if (!std::isnan(obj) && obj_prev-obj >= step_size * t) break;
                    step_size *= armijo_tau;
                    z = z_prev - step_size * inv_hess_grad;
                    obj = objective_f(z);
                }
                if (ct >= armijo_max_iters) throw std::runtime_error("armijo max iterations reached!");
            }

            // invariance
            grad_f(z, grad);

            // check convergence
            if (i > 0 && std::abs(((grad - grad_prev) * (z - z_prev)).sum()) < tol) break;
        }
    }
};

template <class ValueType>
class QPBarrierCQN: public QPBarrierBase<ValueType>
{
    using base_t = QPBarrierBase<ValueType>;
public:
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using rowmat_value_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using colmat_value_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;
    using map_crowmat_value_t = Eigen::Map<const rowmat_value_t>;
    using map_rowmat_value_t = Eigen::Map<rowmat_value_t>;

    const map_crowmat_value_t quad;
    const map_cvec_value_t linear;
    const map_cvec_value_t signs;
    const value_t lmda;
    const size_t n_threads;
    map_rowmat_value_t quad_chol;
    vec_value_t buffer;

    explicit QPBarrierCQN(
        const Eigen::Ref<const rowmat_value_t> quad,
        const Eigen::Ref<const vec_value_t>& linear,
        const Eigen::Ref<const vec_value_t>& signs,
        value_t lmda,
        size_t n_threads,
        Eigen::Ref<rowmat_value_t> quad_chol,
        size_t max_iters,
        value_t tol,
        bool armijo,
        size_t armijo_max_iters,
        value_t armijo_c,
        value_t armijo_tau
    ):
        base_t(max_iters, tol, armijo, armijo_max_iters, armijo_c, armijo_tau),
        quad(quad.data(), quad.rows(), quad.cols()),
        linear(linear.data(), linear.size()),
        signs(signs.data(), signs.size()),
        lmda(lmda),
        n_threads(n_threads),
        quad_chol(quad_chol.data(), quad_chol.rows(), quad_chol.cols()),
        buffer(2 * quad.rows())
    {
        const auto d = quad.rows();
        if (quad.rows() != quad.cols()) {
            throw std::runtime_error("quad must be a square matrix.");
        }
        if (linear.size() != d) {
            throw std::runtime_error("linear must have same size as quad.rows().");
        }
        if (signs.size() != d) {
            throw std::runtime_error("signs must have same size as quad.rows().");
        }
        if (lmda < 0) {
            throw std::runtime_error("lmda must be >= 0.");
        }
        if (n_threads < 1) {
            throw std::runtime_error("n_threads must be >= 1.");
        }
        if (quad_chol.rows() != d || quad_chol.cols() != d) {
            throw std::runtime_error("quad_chol must have the same shape as quad.");
        }
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>& z,
        Eigen::Ref<vec_value_t> out
    ) 
    {
        auto outm = out.matrix();
        adelie_core::matrix::dgemv(quad, z.matrix(), n_threads, buffer /* not used */, outm);
        out += linear - lmda / z;
    }

    void inv_hess_grad(
        const Eigen::Ref<const vec_value_t>& z,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> out
    ) 
    {
        const auto d = z.size();
        Eigen::Map<vec_value_t> H_barrier_diag(buffer.data(), d);
        Eigen::Map<vec_value_t> y(buffer.data() + d, d);

        H_barrier_diag = std::sqrt(lmda) / z.abs();

        quad_chol.diagonal().array() += H_barrier_diag;

        auto L = quad_chol.template triangularView<Eigen::Lower>();
        y.matrix().transpose() = L.solve(grad.matrix().transpose()); 
        out.matrix().transpose() = L.transpose().solve(y.matrix().transpose());

        quad_chol.diagonal().array() -= H_barrier_diag;
    }

    value_t objective(
        const Eigen::Ref<const vec_value_t>& z
    ) 
    {
        const auto d = z.size();
        Eigen::Map<vec_value_t> Hz(buffer.data(), d);
        auto Hzm = Hz.matrix();
        adelie_core::matrix::dgemv(quad, z.matrix(), n_threads, buffer /* not used */, Hzm);
        return (
            0.5 * Hzm.dot(z.matrix())
            + (linear * z).sum()
            - lmda * (signs * z).log().sum()
        );
    }

    vec_value_t solve()
    {
        vec_value_t z = signs;

        base_t::solve(
            z,
            [&](const auto& z, auto& out) { gradient(z, out); },
            [&](const auto& z, const auto& grad, auto& out) { inv_hess_grad(z, grad, out); },
            [&](const auto& z) { return objective(z); }
        );

        return z;
    }
};

} // namespace optimization
} // namespace choosi_core