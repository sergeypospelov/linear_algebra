//
// Created by pospelov on 13.12.2020.
//

#ifndef LINEAR_CORE_MATRIX_HPP_
#define LINEAR_CORE_MATRIX_HPP_

#include <exception>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <vector>

namespace Linear {

template<class T = double>
struct Matrix {
  std::vector<std::vector<T>> a;

  int n;

  explicit Matrix(int n) : n(n) {
    a.resize(n, std::vector<T>(n));
  }
  Matrix(const std::initializer_list<std::initializer_list<T>> &lst) : n(lst.size()) {
    a.resize(n);
    for (auto &inner : lst) {
      if (inner.size() != n) {
        throw std::runtime_error("Bad initializer!");
      }
    }
    auto it = lst.begin();
    for (int i = 0; i < n; i++, it++) {
      for (const T &j : *it) {
        a[i].emplace_back(j);
      }
    }
  }

  Matrix<T> operator*(const Matrix<T> &other) const {
    if (n != other.n) {
      throw std::runtime_error("Matrices have different sizes!");
    }
    Matrix<T> res(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          res[i][j] += a[i][k] * other[k][j];
        }
      }
    }
    return res;
  }
  Matrix<T> operator*=(const Matrix<T> &other) {
    *this = *this * other;
    return *this;
  }

  Matrix<T> operator+(const Matrix<T> &other) const {
    if (n != other.n) {
      throw std::runtime_error("Matrices have different sizes!");
    }
    Matrix<T> res = *this;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res[i][j] += other[i][j];
      }
    }
    return res;
  }
  Matrix<T> operator+=(const Matrix<T> &other) {
    *this = *this + other;
    return *other;
  }

  Matrix<T> operator-() const {
    Matrix res(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res[i][j] = -a[i][j];
      }
    }
    return res;
  }

  Matrix<T> operator-(const Matrix<T> &other) const {
    return *this + (-other);
  }
  Matrix<T> operator-=(const Matrix<T> &other) {
    *this = *this - other;
    return *other;
  }

  Matrix<T> operator*(const T &k) const {
    Matrix<T> res = *this;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res[i][j] *= k;
      }
    }
    return res;
  }
  Matrix<T> operator*=(const T &k) const {
    *this = *this * k;
    return this;
  }

  std::vector<T> operator*(const std::vector<T> &k) const {
    if (n != k.size()) {
      std::cerr << n << " " << k.size() << "\n";
      throw std::runtime_error("Matrix and vector have incompatible dimensions!");
    }
    std::vector<T> res(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res[i] += a[i][j] * k[j];
      }
    }
    return res;
  }

  std::vector<T> &operator[](int i) {
    return a[i];
  }
  const std::vector<T> &operator[](int i) const {
    return a[i];
  }

  Matrix<T> transpose() const {
    Matrix<T> res(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res[i][j] = a[j][i];
      }
    }
    return res;
  }

  bool operator ==(const Matrix<T> &other) const {
    return a == other.a;
  }

  auto begin() const {
    return a.begin();
  }

  auto end() const {
    return a.end();
  }

  size_t size() const {
    return n;
  }


  template<class T2>
  friend std::ostream &operator<<(std::ostream &out, const Matrix<T2> &m);
  template<class T2>
  friend std::istream &operator>>(std::istream &in, Matrix<T2> &m);
};

template<class T>
std::ostream &operator<<(std::ostream &out, const Matrix<T> &m) {
  for (auto &i : m) {
    for (auto &j : i) {
      out << j << " ";
    }
    out << "\n";
  }
  return out;
}

template<class T>
std::istream &operator>>(std::istream &in, Matrix<T> &m) {
  int n;
  in >> n;
  m = Matrix<T>(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      in >> m[i][j];
    }
  }
  return in;
}

}

#endif//LINEAR_CORE_MATRIX_HPP_
