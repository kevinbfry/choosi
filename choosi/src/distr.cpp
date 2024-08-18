#include "decl.hpp"
#include <adelie_core/util/types.hpp>
#include <unsupported/Eigen/SpecialFunctions>

namespace ad = adelie_core;
namespace py = pybind11;

template <class ZType, class ValueType>
auto compute_weights(
    const ZType& z,
    ValueType s_z,
    ValueType l_z,
    ValueType u_z
)
{   
    const auto s_z_scaled = s_z / M_SQRT1_2;
    return 0.5 * (
        ((z-u_z) / s_z_scaled).erfc()
        - 
        ((z-l_z) / s_z_scaled).erfc()
    ).max(1e-100);
}

/**
 * Computes the CDF given by
 * 
 *      \exp(1/4 + a_z - mu_z) \frac{
 *          \sum_i \xi_i^L 
 *          w(a_z-z_i^L) 
 *          \exp(-(z_i^L-(a_z-mu_z+1/2))^2)
 *      }{
 *          \sum_i \xi_i^H
 *          w(mu_z+z_i^H) 
 *      }
 *      
 */
template <class ValueType>
ValueType compute_cdf(
    ValueType mu_z,
    ValueType a_z,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& z_L,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& w_pool_L,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& z_H,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& xi_H,
    ValueType s_z,
    ValueType l_z,
    ValueType u_z
)
{
    using value_t = ValueType;
    const auto diff = a_z - mu_z;
    const auto numer = (
        w_pool_L
        * (-(z_L - (diff + 0.5)).square() + (0.25 + diff)).exp()
    ).sum();
    const auto denom = (
        xi_H 
        * compute_weights(mu_z + z_H, s_z, l_z, u_z)
    ).sum();
    return numer / (denom + (denom <= 0));
}

/**
 * Computes the index to mu such that the CDF given by compute_cdf()
 * is less than or equal to the level at that mu.
 * 
 * @param level   desired level of CDF.
 * @param lower lower bound on mu.
 * @param upper upper bound on mu.
 * @param s     s value.
 * @param x     a grid of values in increasing order.
 * @param w     non-negative weights corresponding to x.
 */
template <class ValueType>
auto compute_cdf_root(
    ValueType level,
    ValueType lower,
    ValueType upper,
    ValueType a_z,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& z_L,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& w_pool_L,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& z_H,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& xi_H,
    ValueType s_z,
    ValueType l_z,
    ValueType u_z,
    ValueType tol
)
{
    using value_t = ValueType;

    const auto alpha = 1 - level;
    const auto compute_sf = [&](auto mu_z) { 
        return 1-compute_cdf<value_t>(
            mu_z, a_z, z_L, w_pool_L, z_H, xi_H, s_z, l_z, u_z
        ); 
    };

    size_t iters = 0;

    // compute SF at the two bounds and early exit if not sf(lower) < alpha < sf(upper)
    value_t sf_lower = compute_sf(lower);
    if (sf_lower >= alpha) return std::make_tuple(lower, iters);
    value_t sf_upper = compute_sf(upper);
    if (sf_upper <= alpha) return std::make_tuple(upper, iters);

    const auto compute_cand = [&]() {
        return 0.5 * (lower + upper);
    };

    while (std::abs(sf_upper - sf_lower) > tol)
    {
        ++iters;
        const auto mu_cand = compute_cand();
        const auto sf_cand = compute_sf(mu_cand);
        if (std::abs(sf_cand - alpha) <= tol || std::abs(upper-lower) <= tol) break;
        lower = (sf_cand < alpha) ? mu_cand : lower;
        upper = (sf_cand < alpha) ? upper : mu_cand;
    }

    return std::make_tuple(compute_cand(), iters);
}

void register_distr(py::module_& m)
{
    m.def("compute_weights", [](
        const Eigen::Ref<const ad::util::rowvec_type<double>>& z,
        double s_z,
        double l_z,
        double u_z
    ) {
        ad::util::rowvec_type<double> out = compute_weights(z, s_z, l_z, u_z);
        return out;
    });
    m.def("compute_cdf", compute_cdf<double>);
    m.def("compute_cdf_root", compute_cdf_root<double>);
}