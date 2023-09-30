// kaldi-decoder/csrc/faster-decoder.h

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/decoder/faster-decoder.h
#ifndef KALDI_DECODER_CSRC_FASTER_DECODER_H_
#define KALDI_DECODER_CSRC_FASTER_DECODER_H_

#include <limits>
#include <string>
#include <vector>

#include "fst/fst.h"
#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/hash-list.h"
#include "kaldifst/csrc/lattice-weight.h"

namespace kaldi_decoder {

struct FasterDecoderOptions {
  // Decoding beam.  Larger->slower, more accurate.
  float beam;

  // Decoder max active states.  Larger->slower; more accurate
  int32_t max_active;

  // Decoder min active states (don't prune if #active less than this).
  int32_t min_active;

  // Increment used in decoder [obscure setting]
  float beam_delta;

  // Setting used in decoder to control hash behavior
  float hash_ratio;

  /*implicit*/ FasterDecoderOptions(
      float beam = 16.0,
      int32_t max_active = std::numeric_limits<int32_t>::max(),
      int32_t min_active = 20, float beam_delta = 0.5, float hash_ratio = 2.0)
      : beam(beam),
        max_active(max_active),
        min_active(min_active),  // This decoder mostly used for
                                 // alignment, use small default.
        beam_delta(beam_delta),
        hash_ratio(hash_ratio) {}

  std::string ToString() const {
    std::ostringstream os;

    os << "FasterDecoderOptions(";
    os << "beam=" << beam << ", ";
    os << "max_active=" << max_active << ", ";
    os << "min_active=" << min_active << ", ";
    os << "beam_delta=" << beam_delta << ", ";
    os << "hash_ratio=" << hash_ratio << ")";

    return os.str();
  }
};

class FasterDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  FasterDecoder(const fst::Fst<fst::StdArc> &fst,
                const FasterDecoderOptions &config);

  FasterDecoder(const FasterDecoder &) = delete;
  FasterDecoder &operator=(const FasterDecoder &) = delete;

  void SetOptions(const FasterDecoderOptions &config) { config_ = config; }

  ~FasterDecoder() { ClearToks(toks_.Clear()); }

  void Decode(DecodableInterface *decodable);

  /// Returns true if a final state was active on the last frame.
  bool ReachedFinal() const;

  /// GetBestPath gets the decoding traceback. If "use_final_probs" is true
  /// AND we reached a final state, it limits itself to final states;
  /// otherwise it gets the most likely token not taking into account
  /// final-probs. Returns true if the output best path was not the empty
  /// FST (will only return false in unusual circumstances where
  /// no tokens survived).
  bool GetBestPath(fst::MutableFst<fst::LatticeArc> *fst_out,
                   bool use_final_probs = true);

  /// As a new alternative to Decode(), you can call InitDecoding
  /// and then (possibly multiple times) AdvanceDecoding().
  void InitDecoding();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32_t max_num_frames = -1);

  /// Returns the number of frames already decoded.
  int32_t NumFramesDecoded() const { return num_frames_decoded_; }

 protected:
  class Token {
   public:
    Arc arc_;  // contains only the graph part of the cost;
    // we can work out the acoustic part from difference between
    // "cost_" and prev->cost_.
    Token *prev_;
    int32_t ref_count_;
    // if you are looking for weight_ here, it was removed and now we just have
    // cost_, which corresponds to ConvertToCost(weight_).
    double cost_;
    inline Token(const Arc &arc, float ac_cost, Token *prev)
        : arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + arc.weight.Value() + ac_cost;
      } else {
        cost_ = arc.weight.Value() + ac_cost;
      }
    }

    // for epsilon transitions, i.e., non-emitting arcs
    inline Token(const Arc &arc, Token *prev)
        : arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + arc.weight.Value();
      } else {
        cost_ = arc.weight.Value();
      }
    }

    inline bool operator<(const Token &other) const {
      return cost_ > other.cost_;
    }

    inline static void TokenDelete(Token *tok) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        delete tok;
        if (prev == nullptr) {
          return;
        } else {
          tok = prev;
        }
      }
    }
  };

  using Elem = HashList<StateId, Token *>::Elem;

  /// Gets the weight cutoff.  Also counts the active tokens.
  double GetCutoff(Elem *list_head, size_t *tok_count, float *adaptive_beam,
                   Elem **best_elem);

  void PossiblyResizeHash(size_t num_toks);

  // ProcessEmitting returns the likelihood cutoff used.
  // It decodes the frame num_frames_decoded_ of the decodable object
  // and then increments num_frames_decoded_
  double ProcessEmitting(DecodableInterface *decodable);

  // TODO(dan): first time we go through this, could avoid using the queue.
  void ProcessNonemitting(double cutoff);

  // HashList defined in ../hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<StateId, Token *> toks_;

  const fst::Fst<fst::StdArc> &fst_;

  FasterDecoderOptions config_;

  // temp variable used in ProcessNonemitting,
  std::vector<const Elem *> queue_;

  std::vector<float> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.

  // Keep track of the number of frames decoded in the current file.
  int32_t num_frames_decoded_;

  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new
  // file. one is to delete the Token objects in the list; the other is to
  // delete the Elem objects.  toks_.Clear() just clears them from the hash and
  // gives ownership to the caller, who then has to call toks_.Delete(e) for
  // each one.  It was designed this way for convenience in propagating tokens
  // from one frame to the next.
  void ClearToks(Elem *list);
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_FASTER_DECODER_H_
