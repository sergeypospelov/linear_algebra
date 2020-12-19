//
// Created by pospelov on 18.12.2020.
//

#ifndef LINEAR_METHODS_EIGEN_QR_HPP_
#define LINEAR_METHODS_EIGEN_QR_HPP_

#include "givens.hpp"
#include "../core/util.hpp"

namespace Linear {

extern const int ITERS;

template<typename T>
std::optional<std::pair<std::vector<T>, Matrix<T>>> eigen_qr(const Matrix<T> &A, const double EPS = 1e-3) {
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

}


#endif//LINEAR_METHODS_EIGEN_QR_HPP_
