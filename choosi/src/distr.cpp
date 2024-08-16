#include "decl.hpp"
#include <adelie_core/util/types.hpp>

namespace ad = adelie_core;
namespace py = pybind11;

/**
 * Computes the CDF function given by
 * 
 *      \frac{
 *          \sum_{i \leq s} w(x_i) \exp(x_i \mu) 
 *      }{
 *          \sum_{i} w(x_i) \exp(x_i \mu) 
 *      }
 * 
 * @param mu    mu value.
 * @param s     s value.
 * @param x     a grid of values in increasing order.
 * @param w     non-negative weights corresponding to x.
 */
template <class ValueType>
ValueType compute_cdf(
    ValueType mu,
    size_t s,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& x,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& w
)
{
    const auto n = x.size();
    const auto shift = (mu > 0) ? x[n-1] : x[0];
    const auto expr = w * (mu * (x - shift)).exp();
    const auto numer = expr.head(s).sum();
    const auto denom = numer + expr.tail(n-s).sum();
    return numer / (denom + (denom <= 0));
}

/**
 * Computes the index to mu such that the CDF given by compute_cdf()
 * is less than or equal to the level at that mu.
 * 
 * @param level   desired level of CDF.
 * @param mu    a grid of mu values in increasing order.
 * @param s     s value.
 * @param x     a grid of values in increasing order.
 * @param w     non-negative weights corresponding to x.
 */
template <class ValueType>
size_t compute_cdf_root(
    ValueType level,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& mu,
    size_t s,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& x,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& w
)
{
    const auto alpha = 1 - level;
    const auto sf = [&](auto m) { 
        return 1-compute_cdf<ValueType>(m, s, x, w); 
    };

    // binary search to find the first mu[i] such that sf(mu[i]) >= alpha.
    size_t begin = 0;
    size_t count = mu.size();
    while (count > 0)
    {
        const auto step = count >> 1;
        const auto cand = begin + step;
        if (sf(mu[cand]) < alpha) {
            begin = cand + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }

    return begin;
}

void register_distr(py::module_& m)
{
    m.def("compute_cdf", compute_cdf<double>);
    m.def("compute_cdf_root", compute_cdf_root<double>);
}