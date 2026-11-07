// kaldi-decoder/csrc/decodable-matrix.h
//
// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/decoder/decodable-matrix.h

/**
 * @file decodable-matrix.h
 * @brief Matrix-backed decodables: precomputed scores as DecodableInterface.
 *
 * This file is copied/modified from @kfile{src/decoder/decodable-matrix.h}.
 * These classes wrap a matrix of per-frame log-likelihoods as a
 * DecodableInterface, optionally mapping transition-ids to pdf-ids through a
 * TransitionInformation.  They are the efficient alternative to a per-call
 * custom decodable when the score matrix is already available: every
 * LogLikelihood() call is an O(1) matrix access.
 *
 * The matrix layout matches kaldi's: row t holds the per-pdf (or per-index)
 * log-likelihoods of frame t,
 * \f[ L \in \mathbb{R}^{T \times P}, \qquad
 *    \mathrm{LogLikelihood}(t, i) = s \cdot L_{t,\,\pi(i)} \f]
 * where \f$\pi\f$ is the tid->pdf map and \f$s\f$ the acoustic scale.
 *
 * Adaptations to this port:
 * - kaldi's Matrix<BaseFloat> is this repo's FloatMatrix (Eigen row-major)
 * - kaldi's MatrixBase/SubMatrix overloads are simplified to FloatMatrix
 * - the raw look-aside pointer trick of kaldi's non-KALDI_PARANOID build is
 *   replaced by direct Eigen coefficient access (identical semantics)
 *
 * Common ownership rules:
 * - The owning constructors copy/keep the matrix and delete it in the
 *   destructor; the borrowing constructors never free it
 * - The decodable itself is owned by the caller (decoders only borrow it)
 *
 * General usage pattern:
 * 1. Compute a (num_frames, num_pdfs) score matrix
 * 2. Wrap it in the decodable that matches your mapping/scale needs
 * 3. Hand the decodable to any decoder (it is a DecodableInterface)
 *
 * @code
 *   kaldi_decoder::FloatMatrix likes(100, 11);  // 100 frames x 11 pdfs
 *   // ... fill likes ...
 *   kaldi_decoder::DecodableMatrixScaledMapped dec(tm, likes, 0.083f);
 *   float s = dec.LogLikelihood(0, 1);  // scale * likes(0, pdf of tid 1)
 * @endcode
 *
 * @see DecodableInterface, TransitionInformation
 */

#ifndef KALDI_DECODER_CSRC_DECODABLE_MATRIX_H_
#define KALDI_DECODER_CSRC_DECODABLE_MATRIX_H_

#include <vector>

#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/eigen.h"
#include "kaldi-decoder/csrc/log.h"
#include "kaldi-decoder/csrc/transition-information.h"

namespace kaldi_decoder {

/// @brief Matrix-backed decodable with acoustic scale and tid->pdf mapping.
///
/// This is kaldi's DecodableMatrixScaledMapped
/// (@kfile{src/decoder/decodable-matrix.h}).  The matrix must have
/// `tm.NumPdfs()` columns; row t holds the per-pdf log-likelihoods of frame
/// t.  Lookups use one-based transition-ids:
/// [ \mathrm{LogLikelihood}(t, \mathit{tid})
///      = s \cdot L_{t,\,\pi(\mathit{tid})} ]
/// with $s$ the scale and $\pi$ the tid->pdf map.
///
/// @code
///   kaldi_decoder::DecodableMatrixScaledMapped dec(tm, likes, 0.083f);
///   float s = dec.LogLikelihood(0, 1);
/// @endcode
///
/// @par Used by: batch decoding of GMM/NN systems where the full score
/// matrix is already available (see also DecodableMatrixMapped,
/// DecodableMatrixScaled).
class DecodableMatrixScaledMapped : public DecodableInterface {
 public:
  /// @brief Construct borrowing the matrix (not deleted when done).
  /// @param [in] tm     Transition model (tid->pdf map).
  /// @param [in] likes  Per-pdf scores, `tm.NumPdfs()` columns.
  /// @param [in] scale  Acoustic scale $s$.
  DecodableMatrixScaledMapped(const TransitionInformation &tm,
                              const FloatMatrix &likes, float scale)
      : trans_model_(tm),
        likes_(&likes),
        tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
        scale_(scale),
        delete_likes_(false) {
    if (likes.cols() != tm.NumPdfs())
      KALDI_DECODER_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                        << likes.cols() << " cols but transition-model has "
                        << tm.NumPdfs() << " pdf-ids.";
  }

  /// @brief Construct owning the matrix (deleted when done).
  /// @param [in] tm     Transition model (tid->pdf map).
  /// @param [in] scale  Acoustic scale $s$.
  /// @param [in] likes  Heap-allocated scores, ownership transferred.
  DecodableMatrixScaledMapped(const TransitionInformation &tm, float scale,
                              const FloatMatrix *likes)
      : trans_model_(tm),
        likes_(likes),
        tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
        scale_(scale),
        delete_likes_(true) {
    if (likes->cols() != tm.NumPdfs())
      KALDI_DECODER_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                        << likes->cols() << " cols but transition-model has "
                        << tm.NumPdfs() << " pdf-ids.";
  }

  /// @return The number of frames (matrix rows).
  int32_t NumFramesReady() const override { return likes_->rows(); }

  bool IsLastFrame(int32_t frame) const override {
    KALDI_DECODER_ASSERT(frame < NumFramesReady());
    return frame == NumFramesReady() - 1;
  }

  /// @brief Scaled score of a transition-id at a frame.
  /// @param [in] frame  Frame index, 0-based.
  /// @param [in] tid    Transition-id, 1-based.
  /// @return $s \cdot L_{t,\pi(\mathit{tid})}$.
  float LogLikelihood(int32_t frame, int32_t tid) override {
    KALDI_DECODER_PARANOID_ASSERT(
        tid >= 1 && tid < static_cast<int32_t>(tid_to_pdf_.size()));
    KALDI_DECODER_PARANOID_ASSERT(frame >= 0 && frame < NumFramesReady());
    return scale_ * (*likes_)(frame, tid_to_pdf_[tid]);
  }

  /// @return The number of transition-ids (indices are one-based).
  int32_t NumIndices() const override {
    return trans_model_.NumTransitionIds();
  }

  /// Destructor; deletes the matrix if this object owns it.
  ~DecodableMatrixScaledMapped() override {
    if (delete_likes_) delete likes_;
  }

 private:
  /// Transition model (for the tid to pdf mapping).
  const TransitionInformation &trans_model_;
  /// The score matrix (borrowed or owned).
  const FloatMatrix *likes_;
  /// Cached reference to the tid->pdf map.
  const std::vector<int32_t> &tid_to_pdf_;
  /// Acoustic scale $s$.
  float scale_;
  /// Whether the destructor deletes @c likes_ (owning constructor).
  bool delete_likes_;
  KALDI_DECODER_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaledMapped);
};

/// @brief Matrix-backed decodable with a frame offset (chunked scores).
///
/// This is like DecodableMatrixScaledMapped, but without an acoustic scale
/// and with a frame offset: row 0 of `likes` is frame `frame_offset` of the
/// utterance.  Useful when a neural net emits scores in chunks for different
/// frame ranges.
/// @code
///   kaldi_decoder::DecodableMatrixMapped dec(tm, likes, 3);  // frames 3..
/// @endcode
class DecodableMatrixMapped : public DecodableInterface {
 public:
  /// @brief Construct borrowing the matrix (not deleted when done).
  /// @param [in] tm            Transition model (tid->pdf map).
  /// @param [in] likes         Per-pdf scores (borrowed).
  /// @param [in] frame_offset  Frame index of row 0.
  DecodableMatrixMapped(const TransitionInformation &tm,
                        const FloatMatrix &likes, int32_t frame_offset = 0);

  /// @brief Construct owning the matrix (deleted when done).
  /// @param [in] tm            Transition model (tid->pdf map).
  /// @param [in] likes         Heap-allocated scores, ownership transferred.
  /// @param [in] frame_offset  Frame index of row 0.
  DecodableMatrixMapped(const TransitionInformation &tm,
                        const FloatMatrix *likes, int32_t frame_offset = 0);

  /// @return `frame_offset + likes.rows()`.
  int32_t NumFramesReady() const override;
  /// @return True for the last frame of the utterance.
  bool IsLastFrame(int32_t frame) const override;
  /// @brief Score of a transition-id at a frame (no scale).
  /// @param [in] frame  Frame index, `frame >= frame_offset`.
  /// @param [in] tid    Transition-id, 1-based.
  /// @return $L_{t - o,\, \pi(\mathit{tid})}$ with offset $o$.
  float LogLikelihood(int32_t frame, int32_t tid) override;
  /// @return The number of transition-ids (indices are one-based).
  int32_t NumIndices() const override;

  /// Destructor; deletes the matrix if this object owns it.
  ~DecodableMatrixMapped() override;

 private:
  /// Transition model (for the tid to pdf mapping).
  const TransitionInformation &trans_model_;
  /// Cached reference to the tid->pdf map.
  const std::vector<int32_t> &tid_to_pdf_;
  /// The score matrix (borrowed or owned).
  const FloatMatrix *likes_;
  /// Non-null only when this object owns @c likes_ (deleted in dtor).
  const FloatMatrix *likes_to_delete_;
  /// Frame index of row 0.
  int32_t frame_offset_ = 0;
  KALDI_DECODER_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixMapped);
};

/// @brief Incrementally-fed matrix decodable for streaming decoding.
///
/// Log-likes live in a matrix that grows via AcceptLoglikes(); the
/// frame-index of the first row is the current offset.  Intended for
/// multi-threaded decoding (mutexes are the caller's job).  Querying a frame
/// below FirstAvailableFrame() is an error -- those rows are gone.
///
/// @code
///   DecodableMatrixMappedOffset dec(tm);
///   dec.AcceptLoglikes(&chunk1, 0);
///   dec.AcceptLoglikes(&chunk2, 1);  // drop 1 oldest frame
///   dec.InputIsFinished();
/// @endcode
///
/// See also DecodableMatrixMapped, which reconstructs the object per chunk
/// instead of mutating one.
class DecodableMatrixMappedOffset : public DecodableInterface {
 public:
  explicit DecodableMatrixMappedOffset(const TransitionInformation &tm)
      : trans_model_(tm),
        tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
        frame_offset_(0),
        input_is_finished_(false) {}

  /// this is not part of the generic Decodable interface.
  int32_t FirstAvailableFrame() const { return frame_offset_; }

  /// Logically, this function appends 'loglikes' (interpreted as newly
  /// available frames) to the log-likelihoods stored in the class.
  ///
  /// This function is destructive of the input "loglikes" because it may
  /// under some circumstances do a shallow copy using Swap().  This function
  /// appends loglikes to any existing likelihoods you've previously supplied.
  void AcceptLoglikes(FloatMatrix *loglikes, int32_t frames_to_discard);

  void InputIsFinished() { input_is_finished_ = true; }

  int32_t NumFramesReady() const override {
    return loglikes_.rows() + frame_offset_;
  }

  bool IsLastFrame(int32_t frame) const override {
    KALDI_DECODER_ASSERT(frame < NumFramesReady());
    return frame == NumFramesReady() - 1 && input_is_finished_;
  }

  float LogLikelihood(int32_t frame, int32_t tid) override {
    KALDI_DECODER_PARANOID_ASSERT(
        tid >= 1 && tid < static_cast<int32_t>(tid_to_pdf_.size()));
    KALDI_DECODER_PARANOID_ASSERT(frame >= frame_offset_ &&
                                  frame < NumFramesReady());
    int32_t pdf_id = tid_to_pdf_[tid];
    return loglikes_(frame - frame_offset_, pdf_id);
  }

  // Note: these indices are 1-based.
  int32_t NumIndices() const override {
    return trans_model_.NumTransitionIds();
  }

  // nothing special to do in destructor.
  ~DecodableMatrixMappedOffset() override = default;

 private:
  /// Transition model (for the tid to pdf mapping).
  const TransitionInformation &trans_model_;
  /// Cached reference to the tid->pdf map.
  const std::vector<int32_t> &tid_to_pdf_;
  /// Stored per-pdf scores of the frames still available (owned).
  FloatMatrix loglikes_;
  /// Frame index of row 0 of @c loglikes_.
  int32_t frame_offset_ = 0;
  /// Set by InputIsFinished(); gates IsLastFrame().
  bool input_is_finished_ = false;
  KALDI_DECODER_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixMappedOffset);
};

/// @brief Plain matrix-backed decodable with an acoustic scale (no tid map).
///
/// Indices are the matrix columns directly (one-based): the score of index
/// `i` at frame `t` is `scale * likes(t, i - 1)`.  Use this when the caller
/// already has per-index (not per-pdf) scores, e.g. for CTC-like graphs.
class DecodableMatrixScaled : public DecodableInterface {
 public:
  DecodableMatrixScaled(const FloatMatrix &likes, float scale)
      : likes_(likes), scale_(scale) {}

  int32_t NumFramesReady() const override { return likes_.rows(); }

  bool IsLastFrame(int32_t frame) const override {
    KALDI_DECODER_ASSERT(frame < NumFramesReady());
    return frame == NumFramesReady() - 1;
  }

  // Note, frames are numbered from zero.
  float LogLikelihood(int32_t frame, int32_t index) override {
    if (index > likes_.cols() || index <= 0 || frame < 0 ||
        frame >= likes_.rows())
      KALDI_DECODER_ERR << "Invalid (frame, index - 1) = (" << frame << ", "
                        << index - 1 << ") for matrix of size " << likes_.rows()
                        << " x " << likes_.cols();
    return scale_ * likes_(frame, index - 1);
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  int32_t NumIndices() const override { return likes_.cols(); }

 private:
  /// The score matrix (borrowed; must outlive this object).
  const FloatMatrix &likes_;
  /// Acoustic scale $s$.
  float scale_;
  KALDI_DECODER_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaled);
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_DECODABLE_MATRIX_H_
