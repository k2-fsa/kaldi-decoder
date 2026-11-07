// kaldi-decoder/csrc/simple-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (author: Arnab Ghoshal);
//                      Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation
#ifndef KALDI_DECODER_CSRC_SIMPLE_DECODER_H_
#define KALDI_DECODER_CSRC_SIMPLE_DECODER_H_

#include <unordered_map>

#include "fst/fst.h"
#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/log.h"
#include "kaldifst/csrc/lattice-weight.h"

/**
   @file simple-decoder.h
   @brief Simple Viterbi decoder (kaldi's SimpleDecoder).

   This file is copied/modified from @kfile{src/decoder/simple-decoder.h}.  Keeps one token per FST state; good for small graphs or as a reference decoder.  See kaldi's doc/decoders.dox for the algorithm.

   @see FasterDecoder, LatticeSimpleDecoder, DecodableInterface.
*/

namespace kaldi_decoder {

/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See kaldi's doc/decoders.dox
    for more information.
    @par Used by:
    Small-graph decoding, tests, and as a reference for the faster decoders.
    @see FasterDecoder, LatticeSimpleDecoder, DecodableInterface
 */
class SimpleDecoder {
 public:
  using StdArc = fst::StdArc;
  using StdWeight = StdArc::Weight;
  using Label = StdArc::Label;
  using StateId = StdArc::StateId;

  /// @brief Construct the decoder.
  /// @param [in] fst   Decoding graph; NOT owned -- must outlive the decoder.
  /// @param [in] beam  Decoding beam (larger = slower, more accurate).
  /// @see Decode, GetBestPath
  SimpleDecoder(const fst::Fst<fst::StdArc> &fst, float beam)
      : fst_(fst), beam_(beam) {}

  /// Destructor; frees the internal tokens (not the graph).
  ~SimpleDecoder();
  SimpleDecoder(const SimpleDecoder &) = delete;
  SimpleDecoder &operator=(const SimpleDecoder &) = delete;

  /// @brief Decode this utterance.
  ///
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether a final state was reached.
  ///
  /// @param [in] decodable  Scores; must outlive the decode.
  /// @return True if any traceback is available.
  /// @see ReachedFinal, GetBestPath
  bool Decode(DecodableInterface *decodable);

  /// @brief Whether a final state was active on the last frame.
  /// @return True if a final state was reached.
  /// @see FinalRelativeCost, Decode
  bool ReachedFinal() const;

  /// @brief Get the decoding traceback.
  ///
  /// If @c use_final_probs is true AND we reached a final state, it limits
  /// itself to final states; otherwise it gets the most likely token without
  /// final-probs.  @c fst_out will be empty (Start() == kNoStateId) if
  /// nothing was available due to search error.  If Decode() returned true,
  /// it is safe to assume GetBestPath() returns true.
  ///
  /// @param [out] fst_out         The best path (a linear Lattice FST).
  /// @param [in]  use_final_probs Account for final-probs (default true).
  /// @return True if the output lattice was nonempty.
  /// @see Decode, FinalRelativeCost
  bool GetBestPath(fst::Lattice *fst_out, bool use_final_probs = true) const;

  /// *** The next functions are from the "new interface". ***

  /// @brief How far the best final cost is from the best cost overall.
  ///
  /// Serves the same function as ReachedFinal() with more information: the
  /// difference between the best (final-cost plus cost) of any token on the
  /// final frame and the best cost of any token on the final frame.  Infinity
  /// means no final states were present on the final frame; usually
  /// nonnegative.
  ///
  /// @return The relative final cost (see above).
  /// @see ReachedFinal
  float FinalRelativeCost() const;

  /// @brief Reset for a new utterance.
  ///
  /// Only needed if you drive decoding with AdvanceDecoding(); Decode() does
  /// its own initialization.
  /// @see AdvanceDecoding, Decode
  void InitDecoding();

  /// @brief Decode up to @c max_num_frames more frames.
  ///
  /// Decodes until there are no more frames ready in the decodable object;
  /// if @c max_num_frames >= 0 it decodes no more than that many frames.
  ///
  /// @param [in] decodable       Scores; must outlive the decode.
  /// @param [in] max_num_frames  Frame limit, or -1 for no limit.
  /// @see InitDecoding, NumFramesDecoded
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32_t max_num_frames = -1);

  /// @return The number of frames already decoded.
  /// @see AdvanceDecoding
  int32_t NumFramesDecoded() const { return num_frames_decoded_; }

 private:
  class Token {
   public:
    fst::LatticeArc arc_;  // We use LatticeArc so that we can separately
                           // store the acoustic and graph cost, in case
                           // we need to produce lattice-formatted output.
    Token *prev_;
    int32_t ref_count_;
    double cost_;  // accumulated total cost up to this point.
    Token(const StdArc &arc, float acoustic_cost, Token *prev)
        : prev_(prev), ref_count_(1) {
      arc_.ilabel = arc.ilabel;
      arc_.olabel = arc.olabel;
      arc_.weight = fst::LatticeWeight(arc.weight.Value(), acoustic_cost);
      arc_.nextstate = arc.nextstate;
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + (arc.weight.Value() + acoustic_cost);
      } else {
        cost_ = arc.weight.Value() + acoustic_cost;
      }
    }
    bool operator<(const Token &other) { return cost_ > other.cost_; }

    static void TokenDelete(Token *tok) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        delete tok;
        if (prev == nullptr) {
          return;
        } else {
          tok = prev;
        }
      }
      KALDI_DECODER_ASSERT(tok->ref_count_ > 0);
    }
  };

  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  void ProcessEmitting(DecodableInterface *decodable);

  void ProcessNonemitting();

  /// Tokens of the current frame (state -> best token).
  std::unordered_map<StateId, Token *> cur_toks_;
  /// Tokens of the previous frame (used by ProcessEmitting).
  std::unordered_map<StateId, Token *> prev_toks_;
  /// The decoding graph (borrowed; not owned).
  const fst::Fst<fst::StdArc> &fst_;
  /// The decoding beam: tokens worse than best + beam are pruned.
  float beam_;
  /// Number of frames decoded in the current file (-1 before InitDecoding).
  int32_t num_frames_decoded_ = -1;

  static void ClearToks(std::unordered_map<StateId, Token *> &toks);  // NOLINT

  static void PruneToks(float beam, std::unordered_map<StateId, Token *> *toks);
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_SIMPLE_DECODER_H_
