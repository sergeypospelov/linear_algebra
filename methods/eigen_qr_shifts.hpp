//
// Created by pospelov on 20.12.2020.
//

#ifndef LINEAR_METHODS_QR_SHIFTS_HPP_
#define LINEAR_METHODS_QR_SHIFTS_HPP_

#include "../core/util.hpp"
#include "givens.hpp"

namespace Linear {

extern const int LIMIT;

template<typename T>
int sgn(const T &val) {
  return (is_zero(val) ? 1 : (T(0) < val) - (val < T(0)));
}

template<typename T>
T wilkinson_shift(const T &A, const T &B, const T &C) {// for matrix ((A, B), (B, C))
  T x1 = 0.5 * (-sqrt(A * A - 2 * A * C + 4 * B * B + C * C) + A + C);
  T x2 = 0.5 * (+sqrt(A * A - 2 * A * C + 4 * B * B + C * C) + A + C);
  return (std::abs(x1 - C) < std::abs(x2 - C) ? x1 : x2);
}

template<typename T>
std::optional<std::pair<std::vector<T>, Matrix<T>>> eigen_qr_shift(const Matrix<T> &A, const double EPS = 1e-3) {// A should be tridiagonalized
  int n = A.n;
  Matrix<T> Q = identity(n);
  if (n == 1) {
    return std::optional(std::make_pair(std::vector<T>{A[0][0]}, Q));
  }
  auto cur_A = A;
  std::vector<T> lambdas(n);

  for (int i = n - 1; i > 0; i--) {
    while (true) {
      if (std::abs(cur_A[i][i - 1]) < EPS) {
        lambdas[i] = cur_A[i][i];
        break;
      }
      T shift = wilkinson_shift(cur_A[i - 1][i - 1], cur_A[i][i - 1], cur_A[i][i]);
      auto [Q_new, R_new] = QR_givens_tridiagonalization(cur_A - identity(n) * shift, i + 1);// O(n^2)
      cur_A = R_new * Q_new + identity(n) * shift;
      Q *= Q_new;
    }
  }
  lambdas[0] = cur_A[0][0];

  return std::optional(make_pair(lambdas, Q));
};

}
#endif//LINEAR_METHODS_QR_SHIFTS_HPP_
