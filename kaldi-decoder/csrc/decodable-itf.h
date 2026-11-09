// kaldi-decoder/csrc/decodable-itf.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Mirko Hannemann;  Go Vivace Inc.;
//                2013  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023-2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/itf/decodable-itf.h

#ifndef KALDI_DECODER_CSRC_DECODABLE_ITF_H_
#define KALDI_DECODER_CSRC_DECODABLE_ITF_H_

#include <stdexcept>

namespace kaldi_decoder {

/**
   @file decodable-itf.h
   @brief The scoring interface between acoustic models and the decoders.

   This file is copied/modified from @kfile{src/itf/decodable-itf.h}.  Every
   decoder in this library consumes acoustic scores exclusively through
   DecodableInterface; the library therefore contains no acoustic-model code
   at all.  GMMs, neural nets, CTC posteriors and Python prototypes can all
   drive the same decoders.

   The interface answers exactly one question:
   \f[
      \mathrm{LogLikelihood}(t, i) = \log p\bigl(\text{label } i
         \,\big|\, \text{frame } t\bigr),
   \f]
   where labels are one-based transition-ids (openfst convention: 0 is
   epsilon).  Higher is better; the decoder adds graph costs separately.

   @code
     class MyDecodable : public kaldi_decoder::DecodableInterface {
      public:
       float LogLikelihood(int32_t frame, int32_t index) override {
         return scores_(frame, index - 1);  // labels are one-based
       }
       int32_t NumFramesReady() const override { return scores_.rows(); }
       bool IsLastFrame(int32_t frame) const override {
         return frame == NumFramesReady() - 1;
       }
       int32_t NumIndices() const override { return scores_.cols(); }
     };
   @endcode

   @par Used by:
   SimpleDecoder, FasterDecoder, LatticeSimpleDecoder,
   LatticeFasterDecoderTpl (all decoders), and the utterance-level wrappers
   in decoder-wrappers.h.

   @see DecodableMatrixScaledMapped, DecodableMatrixScaled, DecodableCtc
        for ready-made implementations;
        @kfile{src/decoder/decodable-matrix.h} in kaldi.

   Common ownership rules:
 *
   - Decoders borrow the decodable and the graph; they never free them
   - Matrix-backed decodables copy (and own) their input matrix unless the
     borrowing constructor is used
   - Lattices returned by GetBestPath()/GetRawLattice() are owned by the
     caller
 *
   General usage pattern:
 *
   1. Provide scores through a DecodableInterface implementation
   2. Construct a decoder with the decoding graph
   3. Decode (one-shot Decode() or InitDecoding()/AdvanceDecoding())
   4. Query ReachedFinal()/GetBestPath()/GetRawLattice()
   5. Release the objects you own (graphs/decodables in the caller)
 *
   Runnable examples live in kaldi-decoder/python/tests/ and in the docs
   (usage/decoding.rst, usage/alignment.rst, usage/streaming.rst).
 */

/// @brief Decodables implement this interface to feed scores to decoders.
///
/// Keep the interface small and model-agnostic (kaldi's design): do not
/// assume scores come from a vector of floats, and do not assume the decoder
/// queries frames strictly left-to-right.
///
/// @par Lifetime:
/// Decoders store a pointer/reference to the decodable without owning it.
/// The caller must keep the object alive for the whole decode.
///
/// @see LatticeFasterDecoderTpl for the main consumer.
class DecodableInterface {
 public:
  virtual ~DecodableInterface() = default;

  /// @brief Acoustic log-likelihood of one label at one frame.
  ///
  /// The value is negated inside the decoder (costs are -log-likelihoods),
  /// so higher is better.  Frames are zero-based; labels (``index``) are
  /// one-based.  Verify NumFramesReady() > frame before calling.
  ///
  /// @param [in] frame  Frame index, 0-based.
  /// @param [in] index  Label (transition-id), 1-based.
  /// @return The log-likelihood \f$\log p(i \mid t)\f$.
  virtual float LogLikelihood(int32_t frame, int32_t index) = 0;

  /// @brief Whether this is the last frame with scores.
  ///
  /// Frames are zero-based; IsLastFrame(-1) is false unless the input is
  /// empty.  In online settings prefer NumFramesReady(): this method may
  /// return false while input is pending and true once termination is
  /// decided.
  ///
  /// @param [in] frame  Frame index, 0-based.
  /// @return True if ``frame`` is the last frame of the utterance.
  virtual bool IsLastFrame(int32_t frame) const = 0;

  /// @brief Number of frames currently available.
  ///
  /// May grow over time in streaming setups; the decoder will not query
  /// beyond it.  This (rather than IsLastFrame) is the primary way decoders
  /// know how far they can proceed.
  ///
  /// @return The number of frames with scores available.
  virtual int32_t NumFramesReady() const {
    throw std::runtime_error(
        "NumFramesReady() not implemented for this decodable type.");
  }

  /// @brief Number of labels (acoustic states) in the model.
  ///
  /// Labels are indexed one-based, i.e. from 1 to NumIndices(); this is for
  /// compatibility with openfst, where 0 is epsilon.
  ///
  /// @return The number of valid labels.
  virtual int32_t NumIndices() const = 0;
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_DECODABLE_ITF_H_
