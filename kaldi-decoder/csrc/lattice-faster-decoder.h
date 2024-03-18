// kaldi-decoder/csrc/lattice-faster-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Mirko Hannemann;
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2018  Zhehuai Chen
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/decoder/lattice-faster-decoder.h
#ifndef KALDI_DECODER_CSRC_LATTICE_FASTER_DECODER_H_
#define KALDI_DECODER_CSRC_LATTICE_FASTER_DECODER_H_

#include <limits>
#include <string>

#include "fst/fst.h"
#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/log.h"

namespace kaldi_decoder {

struct LatticeFasterDecoderConfig {
  float beam;
  int32_t max_active;
  int32_t min_active;
  float lattice_beam;
  int32_t prune_interval;
  bool determinize_lattice;  // not inspected by this class... used in
                             // command-line program.
  float beam_delta;
  float hash_ratio;
  // Note: we don't make prune_scale configurable on the command line, it's not
  // a very important parameter.  It affects the algorithm that prunes the
  // tokens as we go.
  float prune_scale;

  // Number of elements in the block for Token and ForwardLink memory
  // pool allocation.
  int32_t memory_pool_tokens_block_size;
  int32_t memory_pool_links_block_size;

  // Most of the options inside det_opts are not actually queried by the
  // LatticeFasterDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeFaster.
  // fst::DeterminizeLatticePhonePrunedOptions det_opts;

  LatticeFasterDecoderConfig(
      float beam = 16.0,
      int32_t max_active = std::numeric_limits<int32_t>::max(),
      int32_t min_active = 200, float lattice_beam = 10.0,
      int32_t prune_interval = 25, bool determinize_lattice = true,
      float beam_delta = 0.5, float hash_ratio = 2.0, float prune_scale = 0.1,
      int32_t memory_pool_tokens_block_size = 1 << 8,
      int32_t memory_pool_links_block_size = 1 << 8)
      : beam(beam),
        max_active(max_active),
        min_active(min_active),
        lattice_beam(lattice_beam),
        prune_interval(prune_interval),
        determinize_lattice(determinize_lattice),
        beam_delta(beam_delta),
        hash_ratio(hash_ratio),
        prune_scale(prune_scale),
        memory_pool_tokens_block_size(memory_pool_tokens_block_size),
        memory_pool_links_block_size(memory_pool_links_block_size) {}
#if 0
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam,
                   "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active,
                   "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active,
                   "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam,
                   "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("prune-interval", &prune_interval,
                   "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register(
        "determinize-lattice", &determinize_lattice,
        "If true, "
        "determinize the lattice (lattice-determinization, keeping only "
        "best pdf-sequence for each word-sequence).");
    opts->Register(
        "beam-delta", &beam_delta,
        "Increment used in decoding-- this "
        "parameter is obscure and relates to a speedup in the way the "
        "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("hash-ratio", &hash_ratio,
                   "Setting used in decoder to "
                   "control hash behavior");
    opts->Register(
        "memory-pool-tokens-block-size", &memory_pool_tokens_block_size,
        "Memory pool block size suggestion for storing tokens (in elements). "
        "Smaller uses less memory but increases cache misses.");
    opts->Register(
        "memory-pool-links-block-size", &memory_pool_links_block_size,
        "Memory pool block size suggestion for storing links (in elements). "
        "Smaller uses less memory but increases cache misses.");
  }
#endif
  std::string ToString() const {
    std::ostringstream os;

    os << "LatticeFasterDecoderConfig(";
    os << "beam=" << beam << ", ";
    os << "max_active=" << max_active << ", ";
    os << "min_active=" << min_active << ", ";
    os << "lattice_beam=" << lattice_beam << ", ";
    os << "prune_interval=" << prune_interval << ", ";
    os << "determinize_lattice=" << (determinize_lattice ? "True" : "False")
       << ", ";

    os << "beam_delta=" << beam_delta << ", ";
    os << "hash_ratio=" << hash_ratio << ", ";
    os << "prune_scale=" << prune_scale << ", ";
    os << "memory_pool_tokens_block_size=" << memory_pool_tokens_block_size
       << ", ";
    os << "memory_pool_links_block_size=" << memory_pool_links_block_size
       << ")";

    return os.str();
  }
  void Check() const {
    KALDI_DECODER_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0 &&
                         min_active <= max_active && prune_interval > 0 &&
                         beam_delta > 0.0 && hash_ratio >= 1.0 &&
                         prune_scale > 0.0 && prune_scale < 1.0);
  }
};

namespace decoder {
// We will template the decoder on the token type as well as the FST type; this
// is a mechanism so that we can use the same underlying decoder code for
// versions of the decoder that support quickly getting the best path
// (LatticeFasterOnlineDecoder, see lattice-faster-online-decoder.h) and also
// those that do not (LatticeFasterDecoder).

// ForwardLinks are the links from a token to a token on the next frame.
// or sometimes on the current frame (for input-epsilon links).
template <typename Token>
struct ForwardLink {
  using Label = fst::StdArc::Label;

  Token *next_tok;      // the next token [or NULL if represents final-state]
  Label ilabel;         // ilabel on arc
  Label olabel;         // olabel on arc
  float graph_cost;     // graph cost of traversing arc (contains LM, etc.)
  float acoustic_cost;  // acoustic cost (pre-scaled) of traversing arc
  ForwardLink *next;    // next in singly-linked list of forward arcs (arcs
                        // in the state-level lattice) from a token.
  ForwardLink(Token *next_tok, Label ilabel, Label olabel, float graph_cost,
              float acoustic_cost, ForwardLink *next)
      : next_tok(next_tok),
        ilabel(ilabel),
        olabel(olabel),
        graph_cost(graph_cost),
        acoustic_cost(acoustic_cost),
        next(next) {}
};

struct StdToken {
  using ForwardLinkT = ForwardLink<StdToken>;
  using Token = StdToken;

  // Standard token type for LatticeFasterDecoder.  Each active HCLG
  // (decoding-graph) state on each frame has one token.

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  float tot_cost;

  // extra_cost is >= 0.  After calling PruneForwardLinks, this equals the
  // minimum difference between the cost of the best path that this link is a
  // part of, and the cost of the absolute best path, under the assumption that
  // any of the currently active states at the decoding front may eventually
  // succeed (e.g. if you were to take the currently active states one by one
  // and compute this difference, and then take the minimum).
  float extra_cost;

  // 'links' is the head of singly-linked list of ForwardLinks, which is what we
  // use for lattice generation.
  ForwardLinkT *links;

  //'next' is the next in the singly-linked list of tokens for this frame.
  Token *next;

  // This function does nothing and should be optimized out; it's needed
  // so we can share the regular LatticeFasterDecoderTpl code and the code
  // for LatticeFasterOnlineDecoder that supports fast traceback.
  inline void SetBackpointer(Token *backpointer) {}

  // This constructor just ignores the 'backpointer' argument.  That argument is
  // needed so that we can use the same decoder code for LatticeFasterDecoderTpl
  // and LatticeFasterOnlineDecoderTpl (which needs backpointers to support a
  // fast way to obtain the best path).
  StdToken(float tot_cost, float extra_cost, ForwardLinkT *links, Token *next,
           Token *backpointer)
      : tot_cost(tot_cost), extra_cost(extra_cost), links(links), next(next) {}
};

struct BackpointerToken {
  using ForwardLinkT = ForwardLink<BackpointerToken>;
  using Token = BackpointerToken;

  // BackpointerToken is like Token but also
  // Standard token type for LatticeFasterDecoder.  Each active HCLG
  // (decoding-graph) state on each frame has one token.

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  float tot_cost;

  // exta_cost is >= 0.  After calling PruneForwardLinks, this equals
  // the minimum difference between the cost of the best path, and the cost of
  // this is on, and the cost of the absolute best path, under the assumption
  // that any of the currently active states at the decoding front may
  // eventually succeed (e.g. if you were to take the currently active states
  // one by one and compute this difference, and then take the minimum).
  float extra_cost;

  // 'links' is the head of singly-linked list of ForwardLinks, which is what we
  // use for lattice generation.
  ForwardLinkT *links;

  //'next' is the next in the singly-linked list of tokens for this frame.
  BackpointerToken *next;

  // Best preceding BackpointerToken (could be a on this frame, connected to
  // this via an epsilon transition, or on a previous frame).  This is only
  // required for an efficient GetBestPath function in
  // LatticeFasterOnlineDecoderTpl; it plays no part in the lattice generation
  // (the "links" list is what stores the forward links, for that).
  Token *backpointer;

  void SetBackpointer(Token *backpointer) { this->backpointer = backpointer; }

  BackpointerToken(float tot_cost, float extra_cost, ForwardLinkT *links,
                   Token *next, Token *backpointer)
      : tot_cost(tot_cost),
        extra_cost(extra_cost),
        links(links),
        next(next),
        backpointer(backpointer) {}
};

}  // namespace decoder

/** This is the "normal" lattice-generating decoder.
    See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
     for more information.

   The decoder is templated on the FST type and the token type.  The token type
   will normally be StdToken, but also may be BackpointerToken which is to
   support quick lookup of the current best path (see
   lattice-faster-online-decoder.h)

   The FST you invoke this decoder which is expected to equal
   Fst::Fst<fst::StdArc>, a.k.a. StdFst, or GrammarFst.  If you invoke it with
   FST == StdFst and it notices that the actual FST type is
   fst::VectorFst<fst::StdArc> or fst::ConstFst<fst::StdArc>, the decoder object
   will internally cast itself to one that is templated on those more specific
   types; this is an optimization for speed.
 */

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_LATTICE_FASTER_DECODER_H_
