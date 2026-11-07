// kaldi-decoder/csrc/decoder-wrappers.h

// Copyright   2014  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023-2026  Xiaomi Corporation

// This file is copied/modified from kaldi/src/decoder/decoder-wrappers.h
#ifndef KALDI_DECODER_CSRC_DECODER_WRAPPERS_H_
#define KALDI_DECODER_CSRC_DECODER_WRAPPERS_H_

#include <string>
#include <vector>

#include "fst/fst.h"
#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/lattice-faster-decoder.h"
#include "kaldi-decoder/csrc/lattice-simple-decoder.h"
#include "kaldi-decoder/csrc/transition-information.h"

/**
   @file decoder-wrappers.h
   @brief Utterance-level alignment and decoding helpers.

   This file is copied/modified from @kfile{src/decoder/decoder-wrappers.h}.
   Where kaldi's binaries (gmm-align-compiled, gmm-latgen-faster) write
   tables, these functions return plain out-params:

   - AlignUtteranceWrapper: forced alignment of a known word sequence;
   - DecodeUtteranceLatticeFaster / DecodeUtteranceLatticeSimple: one
     utterance of decoding;
   - LatticeBestPath: the core of @kbin{lattice-best-path} -- 1-best under
     lm/acoustic scaling.

   @code
     kaldi_decoder::AlignConfig cfg;
     std::vector<int32_t> alignment, words;
     kaldi_decoder::AlignUtteranceWrapper(cfg, "utt", 0.1f, &graph, &decodable,
                                          &num_done, &num_error, &num_retried,
                                          &tot_like, &frame_count,
                                          &alignment, &words);
   @endcode

   @see LatticeFasterDecoderTpl, DecodableInterface, TransitionInformation.
*/

namespace kaldi_decoder {

/// @brief Alignment beam settings (kaldi's gmm-align-compiled flags).
///
/// @par Used by:
/// AlignUtteranceWrapper.
///
/// @see AlignUtteranceWrapper
struct AlignConfig {
  /// Decoding beam used in alignment (kaldi @c --beam; default 200).
  float beam;
  /// Decoding beam for the second try at alignment
  /// (kaldi @c --retry-beam; 0 disables the retry).
  float retry_beam;

  /// If true, do 'careful' alignment, which is better at detecting
  /// alignment failure (involves a loop back to the start of the decoding
  /// graph; kaldi @c --careful).  Modifies the graph.
  bool careful;

  /*implicit*/ AlignConfig(float beam = 200.0,      // NOLINT
                           float retry_beam = 0.0,  // NOLINT
                           bool careful = false)    // NOLINT
      : beam(beam), retry_beam(retry_beam), careful(careful) {}
};

/// AlignUtteranceWapper is a wrapper for alignment code used in training, that
/// is called from many different binaries, e.g. gmm-align, gmm-align-compiled,
/// sgmm-align, etc.  The writers for alignments and words will only be written
/// to if they are open.  The num_done, num_error, num_retried, tot_like and
/// frame_count pointers will (if non-NULL) be incremented or added to, not set,
/// by this function.
void AlignUtteranceWrapper(
    const AlignConfig &config, const std::string &utt,
    float acoustic_scale,  // affects scores written to scores_writer, if
                           // present
    fst::VectorFst<fst::StdArc> *fst,  // non-const in case config.careful ==
                                       // true, we add loop.
    DecodableInterface *decodable,     // not const but is really an input.
    int32_t *num_done, int32_t *num_error, int32_t *num_retried,
    double *tot_like, int64_t *frame_count, std::vector<int32_t> *alignment,
    std::vector<int32_t> *words);

/// This function modifies the decoding graph for what we call "careful
/// alignment".  The problem we are trying to solve is that if the decoding eats
/// up the words in the graph too fast, it can get stuck at the end, and produce
/// what looks like a valid alignment even though there was really a failure.
/// So what we want to do is to introduce, after the final-states of the graph,
/// a "blind alley" with no final-probs reachable, where the decoding can go to
/// get lost.  Our basic idea is to append the decoding-graph to itself using
/// the fst Concat operation; but in order that there should be final-probs at
/// the end of the first but not the second FST, we modify the right-hand
/// argument to the Concat operation so that it has none of the original
/// final-probs, and add a "pre-initial" state that is final.
void ModifyGraphForCarefulAlignment(fst::VectorFst<fst::StdArc> *fst);

/// @brief Decode one utterance with LatticeSimpleDecoder.
///
/// Produces the outputs kaldi's gmm-latgen-simple writes (adapted from
/// @kfile{src/decoder/decoder-wrappers.cc}: the table writers are replaced
/// by out-params).  As in kaldi, `alignments`/`words` come from the
/// DECODE-TIME best path (decoder.GetBestPath), and `*like_ptr` gets its
/// total cost.  If `lat_out` is non-NULL it receives the raw lattice exactly
/// as kaldi writes it: acoustic costs divided by `acoustic_scale` (skipped
/// when `acoustic_scale == 0`).
///
/// @param [in] decoder        Decoded (or to-be-decoded) decoder.
/// @param [in] decodable      Scores.
/// @param [in] trans_model    Transition information.
/// @param [in] utt            Utterance id (for log messages).
/// @param [in] allow_partial  Output partial results if no final state hit.
/// @param [in] acoustic_scale Acoustic scale used in the decode.
/// @param [out] alignments    Decode-time best path (transition-ids).
/// @param [out] words         Decode-time best path (word ids).
/// @param [out] like_ptr      Utterance likelihood on success.
/// @param [out] lat_out       Optional lattice (see above).
/// @return True on success.
bool DecodeUtteranceLatticeSimple(
    LatticeSimpleDecoder &decoder,  // NOLINT not const but is really an input.
    DecodableInterface &decodable,  // NOLINT not const but is really an input.
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, float acoustic_scale,
    std::vector<int32_t> *alignments,
    std::vector<int32_t> *words,
    double *like_ptr,              // utterance likelihood on success.
    fst::Lattice *lat_out = NULL);  // optional; see above.

/// @brief Decode one utterance with LatticeFasterDecoderTpl.
///
/// See DecodeUtteranceLatticeSimple for the output semantics; this is
/// kaldi's DecodeUtteranceLatticeFaster (@kfile{src/decoder/decoder-wrappers.cc})
/// adapted the same way.  Caution: this will only link correctly for the FST
/// types explicitly instantiated in the .cc file: fst::Fst<fst::StdArc>,
/// fst::VectorFst<fst::StdArc> and fst::ConstFst<fst::StdArc>.
///
/// @param [in] decoder        Decoded (or to-be-decoded) decoder.
/// @param [in] decodable      Scores.
/// @param [in] trans_model    Transition information.
/// @param [in] utt            Utterance id (for log messages).
/// @param [in] allow_partial  Output partial results if no final state hit.
/// @param [in] acoustic_scale Acoustic scale used in the decode.
/// @param [out] alignments    Decode-time best path (transition-ids).
/// @param [out] words         Decode-time best path (word ids).
/// @param [out] like_ptr      Utterance likelihood on success.
/// @param [out] lat_out       Optional lattice (see above).
/// @return True on success.
template <typename FST>
bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<FST>
        &decoder,                   // NOLINT not const but is really an input.
    DecodableInterface &decodable,  // NOLINT not const but is really an input.
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, float acoustic_scale,
    std::vector<int32_t> *alignments,
    std::vector<int32_t> *words,
    double *like_ptr,              // utterance likelihood on success.
    fst::Lattice *lat_out = NULL);  // optional; see above.

/// @brief 1-best of a lattice under lm/acoustic scaling.
///
/// The core of kaldi's `lattice-best-path` binary
/// (@kbin{lattice-best-path}): scale the lattice by (lm_scale,
/// acoustic_scale) via fst::ScaleLattice(fst::LatticeScale(...)) and return
/// the 1-best (ShortestPath) as alignment/word sequences,
/// [ rg\min_P \; \lambda_{\mathrm{lm}} G(P)
///       + \lambda_{\mathrm{ac}} A(P) ]
/// where $G$ is the graph cost (Value1) and $A$ the acoustic cost
/// (Value2).  `*like_ptr` gets the negated total cost of the best path.
///
/// Note on ties: kaldi's binary uses CompactLatticeShortestPath, which on
/// exact total-cost ties keeps the first predecessor in topological order;
/// this implementation uses fst::ShortestPath, which breaks equal sums by
/// smaller graph cost (Value1).  Costs are identical; the 1-best may differ
/// on exact ties only.
///
/// Typical scoring use (kaldi's local/score.sh over lmwt/wip grids) is to
/// call this on the lattice returned by DecodeUtteranceLattice* with
/// acoustic_scale=1.0 (that lattice already has acoustic costs divided by
/// the decode acoustic scale) and lm_scale sweeping the lm weights.
///
/// @param [in] lat            Input lattice.
/// @param [in] lm_scale       Weight of the graph cost (kaldi --lm-scale).
/// @param [in] acoustic_scale Weight of the acoustic cost.
/// @param [out] alignments    Best path (transition-ids); empty on failure.
/// @param [out] words         Best path (word ids); empty on failure.
/// @param [out] like_ptr      Negated total cost (optional).
/// @return True on success; false if the lattice has no complete path
///         (outputs are cleared) -- like kaldi's binary, this does not
///         abort a scoring sweep on one bad lattice.
///
/// @code
///   for (float lmwt : {0.5f, 0.75f, 1.0f}) {
///     std::vector<int32_t> ali, words;
///     double like;
///     kaldi_decoder::LatticeBestPath(lat, lmwt, 1.0f, &ali, &words, &like);
///   }
/// @endcode
/// @see DecodeUtteranceLatticeFaster
bool LatticeBestPath(const fst::Lattice &lat, float lm_scale,
                     float acoustic_scale, std::vector<int32_t> *alignments,
                     std::vector<int32_t> *words, double *like_ptr);

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_DECODER_WRAPPERS_H_
