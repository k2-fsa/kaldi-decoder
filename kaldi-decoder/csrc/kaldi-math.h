// kaldi-decoder/csrc/kaldi-math.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Yanmin Qian;
//                      Jan Silovsky;  Saarland University
//                2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/base/kaldi-math.h
#ifndef KALDI_DECODER_CSRC_KALDI_MATH_H_
#define KALDI_DECODER_CSRC_KALDI_MATH_H_

#include <cmath>
#include <limits>

#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-16
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-7f
#endif

// M_LOG_2PI =  log(2*pi)
#ifndef M_LOG_2PI
#define M_LOG_2PI 1.8378770664093454835606594728112
#endif

#define KALDI_ISINF std::isinf
#define KALDI_ISNAN std::isnan

#include "kaldi-decoder/csrc/log.h"

namespace kaldi_decoder {

static const double kMinLogDiffDouble = std::log(DBL_EPSILON);  // negative!
static const float kMinLogDiffFloat = std::log(FLT_EPSILON);    // negative!

#if !defined(_MSC_VER) || (_MSC_VER >= 1700)
inline double Log1p(double x) { return std::log1p(x); }
inline float Log1p(float x) { return std::log1pf(x); }
#else
inline double Log1p(double x) {
  const double cutoff = 1.0e-08;
  if (x < cutoff)
    return x - 0.5 * x * x;
  else
    return std::log(1.0 + x);
}

inline float Log1p(float x) {
  const float cutoff = 1.0e-07;
  if (x < cutoff)
    return x - 0.5 * x * x;
  else
    return std::log(1.0 + x);
}
#endif

// returns log(exp(x) + exp(y)).
inline float LogAdd(float x, float y) {
  float diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffFloat) {
    float res;
    res = x + Log1p(std::exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

// returns log(exp(x) + exp(y)).
inline double LogAdd(double x, double y) {
  double diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffDouble) {
    double res;
    res = x + Log1p(std::exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

/// return abs(a - b) <= relative_tolerance * (abs(a)+abs(b)).
static inline bool ApproxEqual(float a, float b,
                               float relative_tolerance = 0.001) {
  // a==b handles infinities.
  if (a == b) return true;
  float diff = std::abs(a - b);
  if (diff == std::numeric_limits<float>::infinity() || diff != diff)
    return false;  // diff is +inf or nan.
  return (diff <= relative_tolerance * (std::abs(a) + std::abs(b)));
}

// State for thread-safe random number generator
struct RandomState {
  RandomState();
  unsigned seed;
};

// Returns a random integer between 0 and RAND_MAX, inclusive
int32_t Rand(struct RandomState *state = nullptr);

template <class I>
I Gcd(I m, I n) {
  if (m == 0 || n == 0) {
    if (m == 0 && n == 0) {  // gcd not defined, as all integers are divisors.
      KALDI_DECODER_ERR << "Undefined GCD since m = 0, n = 0.";
    }
    return (m == 0 ? (n > 0 ? n : -n) : (m > 0 ? m : -m));
    // return absolute value of whichever is nonzero
  }
  // could use compile-time assertion
  // but involves messing with complex template stuff.
  KALDI_DECODER_ASSERT(std::numeric_limits<I>::is_integer);
  while (1) {
    m %= n;
    if (m == 0) return (n > 0 ? n : -n);
    n %= m;
    if (n == 0) return (m > 0 ? m : -m);
  }
}

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_KALDI_MATH_H_
