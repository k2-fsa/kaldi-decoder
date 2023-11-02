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

namespace kaldi_decoder {

/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See \ref decoders_simple
    for more information.
 */
class SimpleDecoder {
 public:
  using StdArc = fst::StdArc;
  using StdWeight = StdArc::Weight;
  using Label = StdArc::Label;
  using StateId = StdArc::StateId;

  SimpleDecoder(const fst::Fst<fst::StdArc> &fst, float beam)
      : fst_(fst), beam_(beam) {}

  ~SimpleDecoder();
  SimpleDecoder(const SimpleDecoder &) = delete;
  SimpleDecoder &operator=(const SimpleDecoder &) = delete;

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  bool Decode(DecodableInterface *decodable);

  bool ReachedFinal() const;

  // GetBestPath gets the decoding traceback. If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account
  // final-probs. fst_out will be empty (Start() == kNoStateId) if nothing was
  // available due to search error. If Decode() returned true, it is safe to
  // assume GetBestPath will return true. It returns true if the output lattice
  // was nonempty (i.e. had states in it); using the return value is deprecated.
  bool GetBestPath(fst::Lattice *fst_out, bool use_final_probs = true) const;

  /// *** The next functions are from the "new interface". ***

  /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost
  /// plus cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were
  /// present on the final frame.  It will usually be nonnegative.
  float FinalRelativeCost() const;

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  void InitDecoding();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.  If it returns false, then no tokens are alive,
  /// which is a kind of error state.
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32_t max_num_frames = -1);

  /// Returns the number of frames already decoded.
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

  std::unordered_map<StateId, Token *> cur_toks_;
  std::unordered_map<StateId, Token *> prev_toks_;
  const fst::Fst<fst::StdArc> &fst_;
  float beam_;
  // Keep track of the number of frames decoded in the current file.
  int32_t num_frames_decoded_ = -1;

  static void ClearToks(std::unordered_map<StateId, Token *> &toks);  // NOLINT

  static void PruneToks(float beam, std::unordered_map<StateId, Token *> *toks);
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_SIMPLE_DECODER_H_
