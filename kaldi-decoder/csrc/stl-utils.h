// kaldi-decoder/csrc/stl-utils.h
//
// Copyright (c)  2022  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/util/stl-utils.h

#ifndef KALDI_DECODER_CSRC_STL_UTILS_H_
#define KALDI_DECODER_CSRC_STL_UTILS_H_
#include <algorithm>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace kaldi_decoder {

/// Returns true if the vector is sorted.
template <typename T>
inline bool IsSorted(const std::vector<T> &vec) {
  typename std::vector<T>::const_iterator iter = vec.begin(), end = vec.end();
  if (iter == end) return true;
  while (true) {
    typename std::vector<T>::const_iterator next_iter = iter;
    ++next_iter;
    if (next_iter == end) return true;  // end of loop and nothing out of order
    if (*next_iter < *iter) return false;
    iter = next_iter;
  }
}

/// Sorts and uniq's (removes duplicates) from a vector.
template <typename T>
inline void SortAndUniq(std::vector<T> *vec) {
  std::sort(vec->begin(), vec->end());
  vec->erase(std::unique(vec->begin(), vec->end()), vec->end());
}

/// Returns true if the vector is sorted and contains each element
/// only once.
template <typename T>
inline bool IsSortedAndUniq(const std::vector<T> &vec) {
  typename std::vector<T>::const_iterator iter = vec.begin(), end = vec.end();
  if (iter == end) return true;
  while (true) {
    typename std::vector<T>::const_iterator next_iter = iter;
    ++next_iter;
    if (next_iter == end) return true;  // end of loop and nothing out of order
    if (*next_iter <= *iter) return false;
    iter = next_iter;
  }
}

template <class T>
inline void WriteIntegerVector(std::ostream &os, bool binary,
                               const std::vector<T> &v) {
  // Compile time assertion that this is not called with a wrong type.
  static_assert(std::is_integral_v<T>);
  if (binary) {
    char sz = sizeof(T);  // this is currently just a check.
    os.write(&sz, 1);
    int32_t vecsz = static_cast<int32_t>(v.size());
    if (static_cast<size_t>(vecsz) != v.size()) {
      throw std::runtime_error("WriteIntegerVector: v.size() out of range");
    }

    os.write(reinterpret_cast<const char *>(&vecsz), sizeof(vecsz));
    if (vecsz != 0) {
      os.write(reinterpret_cast<const char *>(&(v[0])), sizeof(T) * vecsz);
    }
  } else {
    // focus here is on prettiness of text form rather than
    // efficiency of reading-in.
    // reading-in is dominated by low-level operations anyway:
    // for efficiency use binary.
    os << "[ ";
    typename std::vector<T>::const_iterator iter = v.begin(), end = v.end();
    for (; iter != end; ++iter) {
      if (sizeof(T) == 1)
        os << static_cast<int16_t>(*iter) << " ";
      else
        os << *iter << " ";
    }
    os << "]\n";
  }
  if (os.fail()) {
    throw std::runtime_error("Write failure in WriteIntegerVector.");
  }
}

template <class T>
inline void ReadIntegerVector(std::istream &is, bool binary,
                              std::vector<T> *v) {
  static_assert(std::is_integral_v<T>);
  if (v == nullptr) {
    throw std::runtime_error("ReadIntegerVector: v != nullptr");
  }
  if (binary) {
    int sz = is.peek();
    if (sz == sizeof(T)) {
      is.get();
    } else {  // this is currently just a check.
      std::ostringstream os;
      os << "ReadIntegerVector: expected to see type of size " << sizeof(T)
         << ", saw instead " << sz << ", at file position " << is.tellg();
      throw std::runtime_error(os.str());
    }
    int32_t vecsz = 0;
    is.read(reinterpret_cast<char *>(&vecsz), sizeof(vecsz));
    if (is.fail() || vecsz < 0) {
      goto bad;  // NOLINT -- verbatim kaldi src/base/io-funcs-inl.h
    }

    v->resize(vecsz);

    if (vecsz > 0) {
      is.read(reinterpret_cast<char *>(&((*v)[0])), sizeof(T) * vecsz);
    }
  } else {
    std::vector<T> tmp_v;  // use temporary so v doesn't use extra memory
                           // due to resizing.
    is >> std::ws;
    if (is.peek() != static_cast<int>('[')) {
      std::ostringstream os;
      os << "ReadIntegerVector: expected to see [, saw " << is.peek()
         << ", at file position " << is.tellg();
      throw std::runtime_error(os.str());
    }
    is.get();       // consume the '['.
    is >> std::ws;  // consume whitespace.
    while (is.peek() != static_cast<int>(']')) {
      if (sizeof(T) == 1) {  // read/write chars as numbers.
        int16_t next_t = 0;
        is >> next_t >> std::ws;
        if (is.fail()) {
          goto bad;  // NOLINT -- verbatim kaldi src/base/io-funcs-inl.h
        } else {
          tmp_v.push_back(static_cast<T>(next_t));
        }
      } else {
        T next_t;
        is >> next_t >> std::ws;
        if (is.fail()) {
          goto bad;  // NOLINT -- verbatim kaldi src/base/io-funcs-inl.h
        } else {
          tmp_v.push_back(next_t);
        }
      }
    }
    is.get();    // get the final ']'.
    *v = tmp_v;  // could use std::swap to use less temporary memory, but this
    // uses less permanent memory.
  }
  if (!is.fail()) return;
bad: {
  std::ostringstream os;
  os << "ReadIntegerVector: read failure at file position " << is.tellg();
  throw std::runtime_error(os.str());
}
}

/// Deletes any non-nullptr pointers in the vector v, and sets
/// the corresponding entries of v to nullptr
template <class A>
void DeletePointers(std::vector<A *> *v) {
  if (v == nullptr) {
    throw std::runtime_error("DeletePointers: v != nullptr");
  }
  typename std::vector<A *>::iterator iter = v->begin(), end = v->end();
  for (; iter != end; ++iter) {
    if (*iter != nullptr) {
      delete *iter;
      *iter = nullptr;  // set to nullptr for extra safety.
    }
  }
}

/// A hashing function-object for pairs of ints
template <typename Int1, typename Int2 = Int1>
struct PairHasher {  // hashing function for pair<int>
  size_t operator()(const std::pair<Int1, Int2> &x) const noexcept {
    // 7853 was chosen at random from a list of primes.
    return x.first + x.second * 7853;
  }
  PairHasher() {  // Check we're instantiated with an integer type.
    static_assert(std::is_integral_v<Int1>);
    static_assert(std::is_integral_v<Int2>);
  }
};

/// Returns true if the vector of pointers contains nullptr pointers.
template <class A>
bool ContainsNullPointers(const std::vector<A *> &v) {
  typename std::vector<A *>::const_iterator iter = v.begin(), end = v.end();
  for (; iter != end; ++iter)
    if (*iter == static_cast<A *>(nullptr)) return true;
  return false;
}

/// A hashing function-object for vectors.
template <typename Int>
struct VectorHasher {  // hashing function for vector<Int>.
  size_t operator()(const std::vector<Int> &x) const noexcept {
    size_t ans = 0;
    auto iter = x.begin(), end = x.end();
    for (; iter != end; ++iter) {
      ans *= kPrime;
      ans += *iter;
    }
    return ans;
  }
  VectorHasher() {  // Check we're instantiated with an integer type.
    static_assert(std::is_integral_v<Int>);
  }

 private:
  static const int kPrime = 7853;
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_STL_UTILS_H_
