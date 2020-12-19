//
// Created by pospelov on 16.12.2020.
//

#ifndef LINEAR_CORE_VEC_HPP_
#define LINEAR_CORE_VEC_HPP_

#include <vector>

namespace Linear {

template<typename T>
struct Vec {
  int n;
  std::vector<T> v;
  Vec(int n) : n(n) {
    v.resize(n);
  }
  size_t size() const {
    return n;
  }
};

}
#endif//LINEAR_CORE_VEC_HPP_
