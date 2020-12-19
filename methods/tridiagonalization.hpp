//
// Created by pospelov on 19.12.2020.
//

#ifndef LINEAR_METHODS_TRIDIAGONALIZATION_HPP_
#define LINEAR_METHODS_TRIDIAGONALIZATION_HPP_

#include "../core/util.hpp"
#include "householder.hpp"

namespace Linear {

template<typename T>
std::pair<Matrix<T>, Matrix<T>> tridiagonalization(const Matrix<T> &A) {
  if (!is_symmetric(A)) {
    throw std::runtime_error("Matrix is not symmetric!");
  }
  int n = A.n;
  Matrix<T> A0 = A;
  Matrix<T> Q = identity(n);
  for (int c = 0; c < n; c++) {
    std::vector<T> v(n);
    for (int i = c + 1; i < n; i++) {
      v[i] = A0[i][c];
    }
    if (is_zero(v)) {
      continue;
    }

    HouseholderMatrix H(v, c + 1);
    A0 = (H * (H * A0).transpose()).transpose();
    Q = H * Q;
  }
  return {A0, Q.transpose()};
}

template<typename T>
std::optional<std::pair<std::vector<T>, Matrix<T>>> eigen_qr_tridiagonalization(const Matrix<T> &A, const double EPS = 1e-3) {
  int n = A.n;
  Matrix<T> Q = identity(n);
  auto cur_A = A;
  for (int i = 0; i < LIMIT; i++) {
    auto [Q_new, R_new] = QR_givens(cur_A);
    cur_A = R_new * Q_new;
    Q *= Q_new;
    Matrix<T> Ak = Q.transpose() * A * Q;
    std::vector<std::pair<T, long double>> circles = gershgorin_circles(Ak);
    long double rad = 0;
    for (auto &i : circles) {
      rad = std::max(rad, i.second);
    }
    if (rad < EPS) {
      std::vector<T> lambdas(n);
      for (auto i = 0; i < n; i++) {
        lambdas[i] = circles[i].first;
      }
      return std::optional(make_pair(lambdas, Q));
    }
  }
  return std::nullopt;
}

}// namespace Linear

#endif//LINEAR_METHODS_TRIDIAGONALIZATION_HPP_
