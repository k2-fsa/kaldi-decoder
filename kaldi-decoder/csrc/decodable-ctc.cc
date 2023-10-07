// kaldi-decoder/csrc/decodable-ctc.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/csrc/decodable-ctc.h"

#include <assert.h>

namespace kaldi_decoder {

DecodableCtc::DecodableCtc(const FloatMatrix &log_probs)
    : log_probs_(log_probs) {
  p_ = &log_probs_(0, 0);
  num_rows_ = log_probs_.rows();
  num_cols_ = log_probs_.cols();
}

DecodableCtc::DecodableCtc(const float *p, int32_t num_rows, int32_t num_cols)
    : p_(p), num_rows_(num_rows), num_cols_(num_cols) {}

float DecodableCtc::LogLikelihood(int32_t frame, int32_t index) {
  // Note: We need to use index - 1 here since
  // all the input labels of the H are incremented during graph
  // construction
  assert(index >= 1);

  return *(p_ + frame * num_cols_ + index - 1);
}

int32_t DecodableCtc::NumFramesReady() const { return num_rows_; }

int32_t DecodableCtc::NumIndices() const { return num_cols_; }

bool DecodableCtc::IsLastFrame(int32_t frame) const {
  assert(frame < NumFramesReady());
  return (frame == NumFramesReady() - 1);
}

}  // namespace kaldi_decoder
