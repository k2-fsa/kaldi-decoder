// kaldi-decoder/csrc/decoder-wrappers.cc

// Copyright   2014  Johns Hopkins University (author: Daniel Povey)
// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023-2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/decoder/decoder-wrappers.cc
//

#include "kaldi-decoder/csrc/decoder-wrappers.h"

#include <vector>

#include "kaldi-decoder/csrc/faster-decoder.h"
#include "kaldi-decoder/csrc/log.h"
#include "kaldifst/csrc/fstext-utils.h"
#include "kaldifst/csrc/lattice-utils.h"
#include "fst/shortest-path.h"

namespace kaldi_decoder {

void AlignUtteranceWrapper(
    const AlignConfig &config, const std::string &utt,
    float acoustic_scale,  // affects scores written to scores_writer, if
                           // present
    fst::VectorFst<fst::StdArc> *fst,  // non-const in case config.careful ==
                                       // true, we add loop.
    DecodableInterface *decodable,     // not const but is really an input.
    int32_t *num_done, int32_t *num_error, int32_t *num_retried,
    double *tot_like, int64_t *frame_count, std::vector<int32_t> *alignment,
    std::vector<int32_t> *words) {
  alignment->clear();
  words->clear();

  if ((config.retry_beam != 0 && config.retry_beam <= config.beam) ||
      config.beam <= 0.0) {
    KALDI_DECODER_ERR << "Beams do not make sense: beam " << config.beam
            << ", retry-beam " << config.retry_beam;
  }

  if (fst->Start() == fst::kNoStateId) {
    KALDI_DECODER_WARN << "Empty decoding graph for " << utt;
    if (num_error != nullptr) {
      (*num_error)++;
    }
    return;
  }

  if (config.careful) {
    ModifyGraphForCarefulAlignment(fst);
  }

  FasterDecoderOptions decode_opts;
  decode_opts.beam = config.beam;

  FasterDecoder decoder(*fst, decode_opts);
  decoder.Decode(decodable);

  bool ans = decoder.ReachedFinal();  // consider only final states.

  if (!ans && config.retry_beam != 0.0) {
    if (num_retried != nullptr) {
      (*num_retried)++;
    }

    KALDI_DECODER_WARN << "Retrying utterance " << utt << " with beam "
             << config.retry_beam;

    decode_opts.beam = config.retry_beam;
    decoder.SetOptions(decode_opts);
    decoder.Decode(decodable);
    ans = decoder.ReachedFinal();
  }

  if (!ans) {  // Still did not reach final state.
    KALDI_DECODER_WARN << "Did not successfully decode file " << utt
             << ", len = " << decodable->NumFramesReady();
    if (num_error != nullptr) {
      (*num_error)++;
    }

    return;
  }

  fst::VectorFst<fst::LatticeArc> decoded;  // linear FST.
  decoder.GetBestPath(&decoded);

  if (decoded.NumStates() == 0) {
    KALDI_DECODER_WARN << "Error getting best path from decoder (likely a bug)";
    if (num_error != NULL) {
      (*num_error)++;
    }

    return;
  }

  fst::LatticeWeight weight;

  fst::GetLinearSymbolSequence(decoded, alignment, words, &weight);

  float like = -(weight.Value1() + weight.Value2()) / acoustic_scale;

  if (num_done != nullptr) {
    (*num_done)++;
  }

  if (tot_like != nullptr) {
    (*tot_like) += like;
  }

  if (frame_count != nullptr) {
    (*frame_count) += decodable->NumFramesReady();
  }
}

// see comment in header.
void ModifyGraphForCarefulAlignment(fst::VectorFst<fst::StdArc> *fst) {
  typedef fst::StdArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;
  typedef Arc::Weight Weight;
  StateId num_states = fst->NumStates();
  if (num_states == 0) {
    KALDI_DECODER_WARN << "Empty FST input.";
    return;
  }

  Weight zero = Weight::Zero();
  // fst_rhs will be the right hand side of the Concat operation.
  fst::VectorFst<fst::StdArc> fst_rhs(*fst);
  // first remove the final-probs from fst_rhs.
  for (StateId state = 0; state < num_states; ++state) {
    fst_rhs.SetFinal(state, zero);
  }

  StateId pre_initial = fst_rhs.AddState();
  Arc to_initial(0, 0, Weight::One(), fst_rhs.Start());
  fst_rhs.AddArc(pre_initial, to_initial);
  fst_rhs.SetStart(pre_initial);
  // make the pre_initial state final with probability one;
  // this is equivalent to keeping the final-probs of the first
  // FST when we do concat (otherwise they would get deleted).
  fst_rhs.SetFinal(pre_initial, Weight::One());

  fst::Concat(fst, fst_rhs);
}

bool DecodeUtteranceLatticeSimple(
    LatticeSimpleDecoder &decoder,  // NOLINT not const but is really an input.
    DecodableInterface &decodable,  // NOLINT not const but is really an input.
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, float acoustic_scale,
    std::vector<int32_t> *alignments,
    std::vector<int32_t> *words, double *like_ptr,
    fst::Lattice *lat_out) {
  if (!decoder.Decode(&decodable)) {
    KALDI_DECODER_WARN << "Failed to decode utterance with id " << utt;
    return false;
  }
  alignments->clear();
  words->clear();

  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_DECODER_WARN << "Outputting partial output for utterance " << utt
               << " since no final-state reached\n";
    } else {
      KALDI_DECODER_WARN << "Not producing output for utterance " << utt
               << " since no final-state reached and "
               << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  fst::LatticeWeight weight = fst::LatticeWeight::Zero();

  // Likelihood as in kaldi: total cost of the decode-time best path.
  fst::VectorFst<fst::LatticeArc> decoded;
  if (!decoder.GetBestPath(&decoded))
    KALDI_DECODER_ERR << "Failed to get traceback for utterance " << utt;
  std::vector<int32_t> tmp_alignment;
  std::vector<int32_t> tmp_words;
  fst::GetLinearSymbolSequence(decoded, &tmp_alignment, &tmp_words, &weight);
  likelihood = -(weight.Value1() + weight.Value2());
  *like_ptr = likelihood;

  // As in kaldi (../kaldi/src/decoder/decoder-wrappers.cc): words and
  // alignments come from the DECODE-TIME best path -- this is exactly what
  // gmm-latgen-faster writes to its words/alignment tables.  (lm-scale /
  // word-ins-penalty scoring is a separate step; see LatticeBestPath.)
  *alignments = tmp_alignment;
  *words = tmp_words;

  // Optionally return the lattice exactly as kaldi writes it: with the
  // acoustic costs divided by acoustic_scale (skipped when it is 0).
  if (lat_out != NULL) {
    if (!decoder.GetRawLattice(lat_out))
      KALDI_DECODER_ERR << "Failed to get raw lattice for utterance " << utt;
    fst::Connect(lat_out);
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::LatticeScale(1.0, 1.0 / acoustic_scale), lat_out);
  }
  return true;
}

// Takes care of output.  Returns true on success.
template <typename FST>
bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<FST>
        &decoder,                   // NOLINT not const but is really an input.
    DecodableInterface &decodable,  // NOLINT not const but is really an input.
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, float acoustic_scale,
    std::vector<int32_t> *alignments,
    std::vector<int32_t> *words, double *like_ptr,
    fst::Lattice *lat_out) {
  using fst::VectorFst;

  if (!decoder.Decode(&decodable)) {
    KALDI_DECODER_WARN << "Failed to decode utterance with id " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_DECODER_WARN << "Outputting partial output for utterance " << utt
               << " since no final-state reached\n";
    } else {
      KALDI_DECODER_WARN << "Not producing output for utterance " << utt
               << " since no final-state reached and "
               << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  fst::LatticeWeight weight = fst::LatticeWeight::Zero();

  // Likelihood as in kaldi: total cost of the decode-time best path.
  fst::VectorFst<fst::LatticeArc> decoded;
  if (!decoder.GetBestPath(&decoded))
    KALDI_DECODER_ERR << "Failed to get traceback for utterance " << utt;
  std::vector<int32_t> tmp_alignment;
  std::vector<int32_t> tmp_words;
  fst::GetLinearSymbolSequence(decoded, &tmp_alignment, &tmp_words, &weight);
  likelihood = -(weight.Value1() + weight.Value2());
  *like_ptr = likelihood;

  // As in kaldi (../kaldi/src/decoder/decoder-wrappers.cc): words and
  // alignments come from the DECODE-TIME best path -- this is exactly what
  // gmm-latgen-faster writes to its words/alignment tables.  (lm-scale /
  // word-ins-penalty scoring is a separate step; see LatticeBestPath.)
  *alignments = tmp_alignment;
  *words = tmp_words;

  // Optionally return the lattice exactly as kaldi writes it: with the
  // acoustic costs divided by acoustic_scale (skipped when it is 0).
  if (lat_out != NULL) {
    if (!decoder.GetRawLattice(lat_out))
      KALDI_DECODER_ERR << "Failed to get raw lattice for utterance " << utt;
    fst::Connect(lat_out);
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::LatticeScale(1.0, 1.0 / acoustic_scale), lat_out);
  }
  return true;
}

template bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<fst::Fst<fst::StdArc>> &decoder,  // NOLINT
    DecodableInterface &decodable,                            // NOLINT
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, float acoustic_scale,
    std::vector<int32_t> *alignments,
    std::vector<int32_t> *words, double *like_ptr,
    fst::Lattice *lat_out);

template bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>> &decoder,  // NOLINT
    DecodableInterface &decodable,                                 // NOLINT
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, float acoustic_scale,
    std::vector<int32_t> *alignments,
    std::vector<int32_t> *words, double *like_ptr,
    fst::Lattice *lat_out);

template bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>> &decoder,  // NOLINT
    DecodableInterface &decodable,                                  // NOLINT
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, float acoustic_scale,
    std::vector<int32_t> *alignments,
    std::vector<int32_t> *words, double *like_ptr,
    fst::Lattice *lat_out);

bool LatticeBestPath(const fst::Lattice &lat, float lm_scale,
                     float acoustic_scale, std::vector<int32_t> *alignments,
                     std::vector<int32_t> *words, double *like_ptr) {
  alignments->clear();
  words->clear();
  if (like_ptr != NULL) *like_ptr = 0;
  // This is the core of kaldi's `lattice-best-path`
  // (kaldi/src/latbin/lattice-best-path.cc): ScaleLattice(LatticeScale) then
  // ShortestPath.  Keeping the LatticeWeight (V1/V2) formulation preserves
  // kaldi's tie-breaking (Compare breaks equal sums by smaller Value1).
  fst::Lattice scaled(lat);
  fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &scaled);
  fst::Lattice best_path;
  fst::ShortestPath(static_cast<const fst::Fst<fst::LatticeArc> &>(scaled),
                    &best_path);
  if (best_path.Start() == fst::kNoStateId) {
    // Like kaldi's lattice-best-path: warn and let the caller skip this
    // lattice instead of aborting the whole scoring sweep.
    KALDI_DECODER_WARN << "LatticeBestPath: best-path failed (empty lattice?)";
    return false;
  }
  fst::LatticeWeight weight;
  fst::GetLinearSymbolSequence(best_path, alignments, words, &weight);
  if (like_ptr != NULL) *like_ptr = -(weight.Value1() + weight.Value2());
  return true;
}

}  // namespace kaldi_decoder
