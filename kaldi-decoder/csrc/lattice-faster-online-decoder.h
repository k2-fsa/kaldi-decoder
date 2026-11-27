// kaldi-decoder/csrc/lattice-faster-online-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Mirko Hannemann;
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2018  Zhehuai Chen
// Copyright (c)  2026  Xiaomi Corporation

// This file is copied/modified from
// kaldi/src/decoder/lattice-faster-online-decoder.h

#ifndef KALDI_DECODER_CSRC_LATTICE_FASTER_ONLINE_DECODER_H_
#define KALDI_DECODER_CSRC_LATTICE_FASTER_ONLINE_DECODER_H_

#include <cstdint>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/lattice-faster-decoder.h"
#include "kaldifst/csrc/lattice-weight.h"

/**
   @file lattice-faster-online-decoder.h
   @brief kaldi's LatticeFasterOnlineDecoder: LatticeFasterDecoder plus a
   cheap best-path traceback.

   This file is copied/modified from
   @kfile{src/decoder/lattice-faster-online-decoder.h}.  The tokens carry a
   backpointer to their best predecessor, so the 1-best path can be read
   without building the raw lattice (BestPathEnd() / TraceBackBestPath()).
   kaldi's online2 code uses it for endpointing and for the
   silence-weighted i-vector (OnlineSilenceWeighting), which needs the best
   path after every chunk of audio.
*/

namespace kaldi_decoder {

/** LatticeFasterOnlineDecoderTpl is as LatticeFasterDecoderTpl but also
    supports an efficient way to get the best path (see the function
    BestPathEnd()), which is useful in endpointing and in situations where you
    might want to frequently access the best path.

    This is only templated on the FST type, since the Token type is required
    to be BackpointerToken.
 */
template <typename FST>
class LatticeFasterOnlineDecoderTpl
    : public LatticeFasterDecoderTpl<FST, decoder::BackpointerToken> {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using Token = decoder::BackpointerToken;
  using ForwardLinkT = decoder::ForwardLink<Token>;

  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of
  // 'fst'.
  LatticeFasterOnlineDecoderTpl(const FST &fst,
                                const LatticeFasterDecoderConfig &config)
      : LatticeFasterDecoderTpl<FST, Token>(fst, config) {}

  // This version of the initializer takes ownership of 'fst', and will delete
  // it when this object is destroyed.
  LatticeFasterOnlineDecoderTpl(const LatticeFasterDecoderConfig &config,
                                FST *fst)
      : LatticeFasterDecoderTpl<FST, Token>(config, fst) {}

  LatticeFasterOnlineDecoderTpl(const LatticeFasterOnlineDecoderTpl &) =
      delete;
  LatticeFasterOnlineDecoderTpl &operator=(
      const LatticeFasterOnlineDecoderTpl &) = delete;

  /// Iterator over the best path, walking backwards in time.  `frame` is one
  /// less than you might expect: after TraceBackBestPath() returns an arc
  /// with a nonzero ilabel for frame t, the iterator's frame is t - 1.
  struct BestPathIterator {
    void *tok;
    int32_t frame;
    // note, "frame" is the frame-index of the frame you'll get the
    // transition-id for next time, if you call TraceBackBestPath on this
    // iterator (assuming it's not an epsilon transition).  Note that this
    // is one less than you might reasonably expect, e.g. it's -1 for
    // the nonemitting transitions before the first frame.
    BestPathIterator(void *t, int32_t f) : tok(t), frame(f) {}
    bool Done() const { return tok == nullptr; }
  };

  /// Outputs an FST corresponding to the single best path through the
  /// lattice.  This is quite efficient because it doesn't get the entire
  /// raw lattice and find the best path through it; instead, it uses the
  /// BestPathEnd and BestPathIterator so it basically traces it back through
  /// the lattice.  Returns true if result is nonempty (using the return
  /// status is deprecated, it will become void).  If "use_final_probs" is
  /// true AND we reached the final-state of the graph then it will include
  /// those as final-probs, else it will treat all final-probs as one.
  bool GetBestPath(fst::Lattice *olat, bool use_final_probs = true) const;

  /// This function returns an iterator that can be used to trace back
  /// the best path.  If use_final_probs == true and at least one final state
  /// survived till the end, it will use the final-probs in working out the
  /// best final Token, and will output the final cost to *final_cost (if
  /// non-NULL), else it will use only the forward likelihood, and will put
  /// zero in *final_cost (if non-NULL).
  /// Requires that NumFramesDecoded() > 0.
  BestPathIterator BestPathEnd(bool use_final_probs,
                               float *final_cost = nullptr) const;

  /// This function can be used in conjunction with BestPathEnd() to trace
  /// back the best path one link at a time (e.g. this can be useful in
  /// endpoint detection).  By "link" we mean a link in the graph; not all
  /// links cross frame boundaries, but each time you see a nonzero ilabel
  /// you can interpret that as a frame.  The return value is the updated
  /// iterator.  It outputs the ilabel and olabel, and the (graph and
  /// acoustic) weight to the "arc" pointer, while leaving its "nextstate"
  /// variable unchanged.
  BestPathIterator TraceBackBestPath(BestPathIterator iter,
                                     fst::LatticeArc *arc) const;

  /// Behaves the same as GetRawLattice but only processes tokens whose
  /// extra_cost is smaller than the best-cost plus the specified beam.
  /// It is only worthwhile to call this function if beam is less than
  /// the lattice_beam specified in the config; otherwise, it would
  /// return essentially the same thing as GetRawLattice, but more slowly.
  bool GetRawLatticePruned(fst::Lattice *ofst, bool use_final_probs,
                           float beam) const;
};

typedef LatticeFasterOnlineDecoderTpl<fst::StdFst> LatticeFasterOnlineDecoder;

/// [not in kaldi] The transition-ids (input labels) of the current best
/// path, one per decoded frame, in time order (input-epsilon links are
/// skipped).  This is the traceback that kaldi's
/// OnlineSilenceWeighting::ComputeCurrentTraceback() reads; pass the result
/// to kaldi-hmm-gmm's OnlineSilenceWeighting.  Returns an empty vector if no
/// frame was decoded.  Note: after FinalizeDecoding(), use_final_probs must
/// be true (same rule as BestPathEnd()).
template <typename FST>
std::vector<int32_t> GetBestPathTransitionIds(
    const LatticeFasterOnlineDecoderTpl<FST> &decoder, bool use_final_probs);

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_LATTICE_FASTER_ONLINE_DECODER_H_
