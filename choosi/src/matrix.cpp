#include "decl.hpp"
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/util/types.hpp>

namespace ad = adelie_core;
namespace py = pybind11;

template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index>
void hessian(
    MatrixType& X,
    const Eigen::Ref<const ad::util::rowvec_type<IndexType>>& subset,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& weights,
    size_t n_threads
)
{
    using vec_index_t = ad::util::rowvec_type<IndexType>;
    using vec_value_t = ad::util::rowvec_type<ValueType>;

    // First, write sequential version.
}

void register_matrix(py::module_& m)
{
    using matrix_naive_base_64_t = ad::matrix::MatrixNaiveBase<double>;

    m.def("hessian", hessian<matrix_naive_base_64_t>);
}