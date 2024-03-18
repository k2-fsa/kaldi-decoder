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
  // It copies the input log_probs
  explicit DecodableCtc(const FloatMatrix &log_probs, int32_t offset = 0);

  // It shares the memory with the input array.
  //
  // @param p Pointer to a 2-d array of shape (num_rows, num_cols).
  //          The array should be kept alive as long as this object is still
  //          alive.
  DecodableCtc(const float *p, int32_t num_rows, int32_t num_cols,
               int32_t offset = 0);

  float LogLikelihood(int32_t frame, int32_t index) override;

  int32_t NumFramesReady() const override;

  // Indices are one-based!  This is for compatibility with OpenFst.
  int32_t NumIndices() const override;

  bool IsLastFrame(int32_t frame) const override;

 private:
  // it saves log_softmax output
  FloatMatrix log_probs_;

  const float *p_ = nullptr;  // pointer to a 2-d array
  int32_t num_rows_;          // number of rows in the 2-d array
  int32_t num_cols_;          // number of cols in the 2-d array
  int32_t offset_ = 0;
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_DECODABLE_CTC_H_
