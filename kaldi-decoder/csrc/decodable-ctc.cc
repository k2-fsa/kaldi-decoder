// kaldi-decoder/csrc/decodable-ctc.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/csrc/decodable-ctc.h"

#include <assert.h>

namespace kaldi_decoder {

DecodableCtc::DecodableCtc(const FloatMatrix &feats) : feature_matrix_(feats) {}

float DecodableCtc::LogLikelihood(int32_t frame, int32_t index) {
  // Note: We need to use index - 1 here since
  // all the input labels of the H are incremented during graph
  // construction
  assert(index >= 1);

  return feature_matrix_(frame, index - 1);
}

int32_t DecodableCtc::NumFramesReady() const { return feature_matrix_.rows(); }

int32_t DecodableCtc::NumIndices() const { return feature_matrix_.cols(); }

bool DecodableCtc::IsLastFrame(int32_t frame) const {
  assert(frame < NumFramesReady());
  return (frame == NumFramesReady() - 1);
}

} // namespace kaldi_decoder
