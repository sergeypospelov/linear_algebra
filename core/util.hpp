//
// Created by pospelov on 13.12.2020.
//

#ifndef LINEAR_CORE_UTIL_HPP_
#define LINEAR_CORE_UTIL_HPP_

#include "matrix.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>

namespace Linear {

const double EPS = 1e-6;

template<typename T>
T abs(const std::vector<T> &v) {
  T res = 0;
  for (const T &x : v) {
    res += std::abs(x) * std::abs(x);
  }
  return sqrt(res);
}

template<typename T>
bool is_zero(const std::vector<T> &x) {
  return abs(x) < EPS;
}

template<typename T>
bool is_zero(const T &x) {
  return std::abs(x) < EPS;
}

template<typename T1, typename T2>
bool check_dimension(T1 a, T2 b) {
  return a.size() == b.size();
}

template<typename T>
std::vector<std::pair<T, long double>> gershgorin_circles(const Matrix<T> &A) {
  std::vector<std::pair<T, long double>> res(A.n);
  for (int i = 0; i < A.n; i++) {
    long double radius = 0;
    for (int j = 0; j < A.n; j++) {
      if (i != j) {
        radius += std::abs(A[i][j]);
      }
    }
    res[i] = {A[i][i], radius};
  }
  return res;
}

template<typename T>
std::vector<T> random_vector(int n) {
  std::vector<T> x0(n);
  std::generate(x0.begin(), x0.end(), [n, t = 0] () mutable { t = (3 * t + 1) % n; return t; });
  return x0;
}

template<typename T>
std::vector<T> operator +(const std::vector<T> &a, const std::vector<T> &b) {
  if (!check_dimension(a, b)) {
    throw std::runtime_error("Vectors have different dimensions!");
  }
  std::vector<T> res = a;
  for (int i = 0; i < a.size(); i++) {
    res[i] += b[i];
  }
  return res;
}

template<typename T>
std::vector<T> operator +=(std::vector<T> &a, const std::vector<T> &b) {
  if (!check_dimension(a, b)) {
    throw std::runtime_error("Vectors have different dimensions!");
  }
  a = a + b;
  return a;
}

template<typename T>
std::vector<T> operator -(const std::vector<T> &a, const std::vector<T> &b) {
  if (!check_dimension(a, b)) {
    throw std::runtime_error("Vectors have different dimensions!");
  }
  std::vector<T> res = a;
  for (int i = 0; i < a.size(); i++) {
    res[i] -= b[i];
  }
  return res;
}

template<typename T>
std::vector<T> operator -=(std::vector<T> &a, const std::vector<T> &b) {
  if (!check_dimension(a, b)) {
    throw std::runtime_error("Vectors have different dimensions!");
  }
  a = a - b;
  return a;
}

template<typename T>
std::vector<T> operator *(const std::vector<T> &a, const T &k) {
  std::vector<T> res = a;
  for (int i = 0; i < a.size(); i++) {
    res[i] *= k;
  }
  return res;
}

template<typename T>
std::vector<T> operator /(const std::vector<T> &a, const T &k) {
  std::vector<T> res = a;
  for (int i = 0; i < a.size(); i++) {
    res[i] /= k;
  }
  return res;
}

template<typename T>
std::vector<T> operator -(const std::vector<T> &a) {
  return a * T(-1);
}

template<typename T>
std::vector<T> operator *=(const std::vector<T> &a, const T &k) {
  a = a * k;
  return a;
}


template<class T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &m) {
  for (auto &i : m) {
    out << i << " ";
  }
  out << "\n";
  return out;
}

template<typename T = double>
Matrix<T> identity(int n) {
  Matrix<T> res(n);
  for (int i = 0; i < n; i++) {
    res[i][i] = 1;
  }
  return res;
}

template<typename T>
Matrix<T> mult(const std::vector<T> &a, const std::vector<T> &b) {
  if (!check_dimension(a, b)) {
    throw std::runtime_error("Bad dimensions!");
  }
  int n = a.size();
  Matrix<T> res(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      res[i][j] = a[i] * b[j];
    }
  }
  return res;
}

template<typename T>
std::vector<T> standart(int n, int i) {
  std::vector<T> v(n);
  v[i] = 1;
  return v;
}

template<typename T>
const std::vector<T> normalize(const std::vector<T> &v) {
  if (is_zero(v)) {
    return std::vector<T>(v.size());
  }
  return v / abs(v);
}

template<typename T>
const T scalar(const std::vector<T> &v1, const std::vector<T> &v2) {
  if (!check_dimension(v1, v2))  {
    throw std::runtime_error("Bad dimensions!");
  }
  T sum = 0;
  for (int i = 0; i < v1.size(); i++) {
    sum += v1[i] * v2[i];
  }
  return sum;
}

template <typename T>
bool is_symmetric(const Matrix<T> &A) {
  int n = A.n;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (!is_zero(A[i][j] - A[j][i])) {
        return false;
      }
    }
  }
  return true;
}


}// namespace Linear

#endif//LINEAR_CORE_UTIL_HPP_
