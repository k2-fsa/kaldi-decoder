// kaldi-decoder/csrc/decodable-ctc.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_DECODER_CSRC_DECODABLE_CTC_H_
#define KALDI_DECODER_CSRC_DECODABLE_CTC_H_

#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/eigen.h"

namespace kaldi_decoder {

class DecodableCtc : public DecodableInterface {
 public:
  explicit DecodableCtc(const FloatMatrix &feats);

  float LogLikelihood(int32_t frame, int32_t index) override;

  int32_t NumFramesReady() const override;

  // Indices are one-based!  This is for compatibility with OpenFst.
  int32_t NumIndices() const override;

  bool IsLastFrame(int32_t frame) const override;

 private:
  // it saves log_softmax output
  FloatMatrix feature_matrix_;
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_DECODABLE_CTC_H_
