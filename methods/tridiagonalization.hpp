//
// Created by pospelov on 19.12.2020.
//

#ifndef LINEAR_METHODS_TRIDIAGONALIZATION_HPP_
#define LINEAR_METHODS_TRIDIAGONALIZATION_HPP_

#include "../core/util.hpp"
#include "givens.hpp"
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
std::pair<Matrix<T>, Matrix<T>> QR_givens_tridiagonalization(const Matrix<T> &A, int Mx = -1) { // A should be tridiagonalized
  int n = A.n;
  Mx = (Mx == -1 ? n : Mx);
  Matrix<T> A0 = A;
  Matrix<T> Q = identity(n);
  for (int c = 0; c < Mx; c++) {
    int r = c;
    while (r < std::min(c + 1, Mx) && is_zero(A0[r][c])) {
      r++;
    }
    if (r == Mx) {
      continue;
    }
    for (int i = r + 1; i < std::min(Mx, c + 2); i++) {
      GivensMatrix G(i, r, A0[i][c], A0[r][c]);
      A0 = G * A0;
      Q = G * Q;
    }
    GivensMatrix G(r, c, 1, 0);
    A0 = G * A0;
    Q = G * Q;
  }
  return {Q.transpose(), A0};
}

template<typename T>
std::optional<std::pair<std::vector<T>, Matrix<T>>> eigen_qr_tridiagonalization(const Matrix<T> &A, const double EPS = 1e-3) {
  int n = A.n;
  Matrix<T> Q = identity(n);
  auto cur_A = A;
  for (int i = 0; i < LIMIT; i++) {
    auto [Q_new, R_new] = QR_givens_tridiagonalization(cur_A); // O(n^2)
    cur_A = R_new * Q_new;
    Q *= Q_new;
    std::vector<std::pair<T, long double>> circles = gershgorin_circles(cur_A); // O(n^2)
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
