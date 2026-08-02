/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

#ifndef NUMERICS_DISTRIBUTED_ARRAY_SLATE_OPS_MATRIX_ARRAY_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_SLATE_OPS_MATRIX_ARRAY_HPP

/*
 * slate_ops overloads for math::nda::distributed_matrix_array.
 *
 * Kept in a separate header so that existing slate_ops users are untouched and call sites can
 * be migrated one at a time.
 *
 * Two things are simpler here than in the distributed_array path:
 *
 *  1. distributed_matrix_array is natively column-major, so its slate::Matrix is the matrix,
 *     not its transpose. None of the "swap A and B / conjugate because the view is
 *     transposed" bookkeeping in slate_ops.hpp is needed, and the `hermitian` template
 *     parameter of lu_solve/least_squares_solve becomes irrelevant (it existed only to make
 *     that trick valid). It is accepted and ignored so call sites need no edit.
 *  2. Tiles are owned by the container in tile-major order with ld == tileMb(i), which is what
 *     slate's batched device path requires, so there is no make_slate view and no stride
 *     hazard.
 */

#include <array>
#include <utility>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "numerics/distributed_array/matrix_array.hpp"
#if defined(ENABLE_SLATE)
#include "slate/slate.hh"
#endif

namespace math::nda
{

/// Minimal concept for the container; deliberately distinct from DistributedArray, which
/// requires local()/Array_t that distributed_matrix_array does not provide.
template<typename A>
concept DistributedMatrixArray = requires(std::decay_t<A>& a, std::decay_t<A> const& ca) {
  { std::decay_t<A>::rank } -> std::convertible_to<int>;
  { ca.n_local_batch() } -> std::convertible_to<long>;
  { ca.global_shape() };
  { ca.grid() };
  { ca.mb() } -> std::convertible_to<long>;
  { ca.nb() } -> std::convertible_to<long>;
  { a.buffer() };
};

#if defined(ENABLE_SLATE)

/***************************************************************************/
/*                            operand tagging                              */
/***************************************************************************/

/// transpose/dagger wrapper for the new container. The existing math::detail::*_tag types
/// cannot be reused: they require MA::Array_t.
template<typename MA>
struct ma_op_tag {
  MA* a;
  slate::Op op;
};

template<DistributedMatrixArray MA>
auto transpose(MA& a) { return ma_op_tag<MA>{std::addressof(a), slate::Op::Trans}; }
template<DistributedMatrixArray MA>
auto dagger(MA& a) { return ma_op_tag<MA>{std::addressof(a), slate::Op::ConjTrans}; }
template<DistributedMatrixArray MA>
auto H(MA& a) { return dagger(a); }

namespace detail
{
template<DistributedMatrixArray MA> MA& ma_array(MA& a) { return a; }
template<typename MA> MA& ma_array(ma_op_tag<MA> const& t) { return *t.a; }

template<DistributedMatrixArray MA> slate::Op ma_op(MA const&) { return slate::Op::NoTrans; }
template<typename MA> slate::Op ma_op(ma_op_tag<MA> const& t) { return t.op; }

/// apply an Op flag to a slate matrix (all three branches share the return type)
template<typename T>
slate::Matrix<T> apply_op(slate::Matrix<T> A, slate::Op op) {
  if (op == slate::Op::Trans)     return slate::transpose(A);
  if (op == slate::Op::ConjTrans) return slate::conj_transpose(A);
  return A;
}

template<typename MA>
slate::Options slate_opts() {
  if constexpr (std::decay_t<MA>::on_host) {
    return slate::Options{
#if defined(USE_SLATE_HOSTBATCH)
      { slate::Option::Target, slate::Target::HostBatch }
#endif
    };
  } else {
    return slate::Options{ { slate::Option::Target, slate::Target::Devices },
                           { slate::Option::Lookahead, 1 } };
  }
}

/// batch consistency: all operands must share the batch decomposition
template<typename A_t, typename B_t>
void check_batch(A_t const& A, B_t const& B, std::string_view who) {
  utils::check(A.n_local_batch() == B.n_local_batch(),
      "{}: local batch count mismatch: {} vs {}", who, A.n_local_batch(), B.n_local_batch());
  utils::check(*A.communicator() == *B.communicator(), "{}: communicator mismatch", who);
}
} // namespace detail

namespace slate_ops
{

/***************************************************************************/
/*                                  gemm                                   */
/***************************************************************************/

/*
 * C = alpha * op(A) * op(B) + beta * C, for every local batch element.
 * A and B may be wrapped in transpose()/dagger().
 */
template<typename T, typename A_t, typename B_t, DistributedMatrixArray C_t>
auto& multiply(T alpha, A_t&& A, B_t&& B, T beta, C_t&& C)
{
  auto& dA = ::math::nda::detail::ma_array(A);
  auto& dB = ::math::nda::detail::ma_array(B);
  auto& dC = C;
  auto opA = ::math::nda::detail::ma_op(A);
  auto opB = ::math::nda::detail::ma_op(B);

  ::math::nda::detail::check_batch(dA, dC, "slate_ops::multiply");
  ::math::nda::detail::check_batch(dB, dC, "slate_ops::multiply");

  auto opts = ::math::nda::detail::slate_opts<C_t>();
  for (long ib = 0; ib < dC.n_local_batch(); ++ib) {
    auto As = ::math::nda::detail::apply_op(dA.slate_matrix(ib), opA);
    auto Bs = ::math::nda::detail::apply_op(dB.slate_matrix(ib), opB);
    auto Cs = dC.slate_matrix(ib);
    utils::check(As.m() == Cs.m() and Bs.n() == Cs.n() and As.n() == Bs.m(),
        "slate_ops::multiply: shape mismatch ({}x{}) * ({}x{}) -> ({}x{})",
        As.m(), As.n(), Bs.m(), Bs.n(), Cs.m(), Cs.n());
    slate::multiply(alpha, As, Bs, beta, Cs, opts);
  }
  return dC;
}

template<typename A_t, typename B_t, DistributedMatrixArray C_t>
auto& multiply(A_t&& A, B_t&& B, C_t&& C)
{
  using T = typename std::decay_t<C_t>::value_type;
  return multiply(T{1.0}, std::forward<A_t>(A), std::forward<B_t>(B), T{0.0},
                 std::forward<C_t>(C));
}

/***************************************************************************/
/*                                 solves                                  */
/***************************************************************************/

/*
 * Solve A X = B in place (B <- X), per local batch element.
 * `hermitian` is accepted and ignored: see the header comment.
 */
template<bool hermitian = false, DistributedMatrixArray A_t, DistributedMatrixArray B_t>
long lu_solve(A_t&& A, B_t&& B)
{
  ::math::nda::detail::check_batch(A, B, "slate_ops::lu_solve");
  utils::check(A.global_shape()[std::decay_t<A_t>::rank-2] ==
               A.global_shape()[std::decay_t<A_t>::rank-1],
               "slate_ops::lu_solve: A must be square.");
  auto opts = ::math::nda::detail::slate_opts<A_t>();
  long info = 0;
  for (long ib = 0; ib < A.n_local_batch(); ++ib) {
    auto As = A.slate_matrix(ib);
    auto Bs = B.slate_matrix(ib);
    long i = slate::lu_solve(As, Bs, opts);
    if (i != 0 and info == 0) info = i;
  }
  return info;
}

template<bool hermitian = false, DistributedMatrixArray A_t, DistributedMatrixArray B_t>
long least_squares_solve(A_t&& A, B_t&& B)
{
  ::math::nda::detail::check_batch(A, B, "slate_ops::least_squares_solve");
  auto opts = ::math::nda::detail::slate_opts<A_t>();
  for (long ib = 0; ib < A.n_local_batch(); ++ib) {
    auto As = A.slate_matrix(ib);
    auto Bs = B.slate_matrix(ib);
    slate::least_squares_solve(As, Bs, opts);
  }
  return 0;
}

/*
 * A <- A^{-1}, per local batch element.
 *
 * Uses the OUT-OF-PLACE getri. The in-place slate::getri is host-only whatever Target is
 * requested (its internal ops are hardcoded Target::HostTask and it reaches tiles through
 * at(i,j,HostNum)); slate's own source says "This routine is in-place and does not support
 * GPUs. There is another one (out-of-place) that does". The out-of-place form is set(I) +
 * getrs, both of which have real device paths.
 */
template<DistributedMatrixArray A_t>
void inverse(A_t&& A)
{
  constexpr int R = std::decay_t<A_t>::rank;
  utils::check(A.global_shape()[R-2] == A.global_shape()[R-1],
               "slate_ops::inverse: A must be square.");
  auto opts = ::math::nda::detail::slate_opts<A_t>();
  for (long ib = 0; ib < A.n_local_batch(); ++ib) {
    auto As = A.slate_matrix(ib);
    slate::Pivots pivots;
    long info = slate::getrf(As, pivots, opts);
    utils::check(info == 0, "slate_ops::inverse: getrf info: {}", info);
    auto Bs = As.emptyLike();
    Bs.insertLocalTiles(std::decay_t<A_t>::on_host ? slate::Target::Host
                                                   : slate::Target::Devices);
    slate::getri(As, pivots, Bs, opts);   // Bs = As^{-1}
    slate::copy(Bs, As, opts);            // back into the caller's buffer
  }
}

} // namespace slate_ops

#endif // ENABLE_SLATE

} // namespace math::nda

#endif
