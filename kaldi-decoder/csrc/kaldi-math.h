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

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_KALDI_MATH_H_
