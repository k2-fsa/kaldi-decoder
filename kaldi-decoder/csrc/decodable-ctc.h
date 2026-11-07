// kaldi-decoder/csrc/decodable-ctc.h

// Copyright (c)  2023-2026  Xiaomi Corporation
//
// Original class (no kaldi counterpart): a matrix-backed decodable for
// CTC-style per-frame log-probabilities.

/**
   @file decodable-ctc.h
   @brief Matrix-backed decodable for CTC-style per-frame log-probs.

   An original class (kaldi has no CTC decodable); in spirit it is
   @kfile{src/decoder/decodable-matrix.h}'s DecodableMatrixScaled without any
   transition-model mapping.  Hand in a (num_frames, num_classes) matrix of
   log-probabilities; the class copies it.

   Indices are one-based (openfst convention): LogLikelihood(t, 1) reads
   column 0.

   @code
     kaldi_decoder::FloatMatrix log_probs(2, 2);
     log_probs << -0.1f, -5.0f, -5.0f, -0.1f;
     kaldi_decoder::DecodableCtc dec(log_probs);
     float s = dec.LogLikelihood(0, 1);  // -0.1
   @endcode

   @see DecodableMatrixScaled for the kaldi-style matrix decodable;
        DecodableInterface for the interface.

   Common ownership rules:
 *
   - Decoders borrow the graph and the decodable; the caller owns them
   - Returned lattices/sequences are owned by the caller
   - Matrix decodables own copies of their input matrices
 *
   See kaldi-decoder/python/tests/ and docs/usage/ for runnable examples.
 */

#ifndef KALDI_DECODER_CSRC_DECODABLE_CTC_H_
#define KALDI_DECODER_CSRC_DECODABLE_CTC_H_

#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/eigen.h"
namespace kaldi_decoder {

/// @brief Decodable over a matrix of CTC log-probabilities.
///
/// Row t holds the per-class log-probs of frame t; the matrix is copied at
/// construction.  Labels are one-based: class c is column c-1.
///
/// @code
///   kaldi_decoder::FloatMatrix log_probs(2, 2);
///   log_probs << -0.1f, -5.0f, -5.0f, -0.1f;
///   kaldi_decoder::DecodableCtc dec(log_probs);
///   float s = dec.LogLikelihood(0, 1);  // -0.1
/// @endcode
///
/// @par Used by:
/// CTC-style decoding with any decoder in this library (the label space is
/// the matrix columns).
///
/// @see DecodableInterface, DecodableMatrixScaled
class DecodableCtc : public DecodableInterface {
 public:
  /// @brief Construct from a log-probability matrix (copied).
  /// @param [in] log_probs  (num_frames, num_classes) scores.
  /// @param [in] offset     Frame index of row 0 (chunked streaming).
  /// @see DecodableCtc(const float *, int32_t, int32_t, int32_t)
  explicit DecodableCtc(const FloatMatrix &log_probs, int32_t offset = 0);

  /// @brief Construct sharing the memory of a raw array (NOT copied).
  ///
  /// The array must stay alive as long as this object; nothing is freed by
  /// the destructor.
  /// @param [in] p          Pointer to a (num_rows, num_cols) array.
  /// @param [in] num_rows   Number of frames in the array.
  /// @param [in] num_cols   Number of classes.
  /// @param [in] offset     Frame index of row 0.
  /// @see DecodableCtc(const FloatMatrix &, int32_t)
  DecodableCtc(const float *p, int32_t num_rows, int32_t num_cols,
               int32_t offset = 0);

  /// @brief Log-prob of one class at one frame.
  /// @param [in] frame  Frame index, 0-based.
  /// @param [in] index  Class index, 1-based (column index-1).
  /// @return The stored log-probability.
  /// @see NumIndices, NumFramesReady
  float LogLikelihood(int32_t frame, int32_t index) override;

  /// @return The number of frames available (matrix rows + offset).
  /// @see LogLikelihood, IsLastFrame
  int32_t NumFramesReady() const override;

  /// @return The number of classes; indices are one-based (openfst).
  /// @see LogLikelihood
  int32_t NumIndices() const override;

  /// @return True for the last frame of the matrix.
  /// @see NumFramesReady
  bool IsLastFrame(int32_t frame) const override;

 private:
  /// Copied matrix of per-frame log-probs (owned); empty when sharing @c p_.
  FloatMatrix log_probs_;

  /// Pointer to a 2-d (num_rows x num_cols) array shared with the caller.
  const float *p_ = nullptr;
  /// Number of rows in the shared array @c p_.
  int32_t num_rows_;
  /// Number of columns in the shared array @c p_.
  int32_t num_cols_;
  /// Frame index of row 0 (chunked streaming).
  int32_t offset_ = 0;

  // p_ aliases the log_probs_ buffer; copying would leave the copy's p_
  // pointing into the source object's matrix (dangling after destruction).
  KALDI_DECODER_DISALLOW_COPY_AND_ASSIGN(DecodableCtc);
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_DECODABLE_CTC_H_
