// kaldi-decoder/csrc/lattice-faster-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Mirko Hannemann;
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2018  Zhehuai Chen
// Copyright (c)  2023-2026  Xiaomi Corporation

// This file is copied/modified from kaldi/src/decoder/lattice-faster-decoder.h

#ifndef KALDI_DECODER_CSRC_LATTICE_FASTER_DECODER_H_
#define KALDI_DECODER_CSRC_LATTICE_FASTER_DECODER_H_

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "fst/fst.h"
#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/determinize-lattice-pruned.h"
#include "kaldi-decoder/csrc/hash-list.h"
#include "kaldifst/csrc/lattice-weight.h"

/**
   @file lattice-faster-decoder.h
   @brief The standard lattice decoder (kaldi's LatticeFasterDecoder).

   This file is copied/modified from
   @kfile{src/decoder/lattice-faster-decoder.h}.  The decoder behind
   gmm-latgen-faster: beam-pruned token passing that keeps near-best
   alternatives as a lattice.  See kaldi's doc/lattices.dox and
   doc/decoders.dox.

   @see LatticeSimpleDecoder, decoder-wrappers.h, DecodableInterface.
*/

namespace kaldi_decoder {

struct LatticeFasterDecoderConfig {
  /// Decoder beam (kaldi @c --beam): which paths stay alive.
  float beam;
  /// Max active tokens per frame (kaldi @c --max_active).
  int32_t max_active;
  /// Don't prune below this many tokens (kaldi @c --min_active).
  int32_t min_active;
  /// Lattice beam (kaldi @c --lattice_beam): how much worse than the best
  /// path an alternative may be to stay in the output lattice.
  float lattice_beam;
  /// Prune every this many frames (kaldi @c --prune_interval).
  int32_t prune_interval;
  /// Whether the CALLER will determinize the output lattice (kaldi
  /// @c determinize_lattice); not inspected by this class.
  bool determinize_lattice;
  /// Adaptive beam increment (kaldi @c beam_delta).
  float beam_delta;
  /// Hash-table sizing ratio (kaldi @c hash_ratio).
  float hash_ratio;
  /// Token-pruning scale; not configurable on the command line
  /// (kaldi @c prune_scale).
  float prune_scale;

  /// Number of elements in the block for Token memory pool allocation
  /// (kaldi @c memory_pool_tokens_block_size).
  int32_t memory_pool_tokens_block_size;
  /// Number of elements in the block for ForwardLink memory pool allocation
  /// (kaldi @c memory_pool_links_block_size).
  int32_t memory_pool_links_block_size;

  // Most of the options inside det_opts are not actually queried by the
  // LatticeFasterDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeFaster.
  DeterminizeLatticePhonePrunedOptions det_opts;

  /// @brief Construct with kaldi's defaults (all arguments optional).
  /// @see Check, ToString
  explicit LatticeFasterDecoderConfig(
      float beam = 16.0,
      int32_t max_active = std::numeric_limits<int32_t>::max(),
      int32_t min_active = 200, float lattice_beam = 10.0,
      int32_t prune_interval = 25, bool determinize_lattice = true,
      float beam_delta = 0.5, float hash_ratio = 2.0, float prune_scale = 0.1,
      int32_t memory_pool_tokens_block_size = (1 << 8),
      int32_t memory_pool_links_block_size = (1 << 8),
      const DeterminizeLatticePhonePrunedOptions &det_opts = {})
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
        memory_pool_links_block_size(memory_pool_links_block_size),
        det_opts(det_opts) {}

  /// @return A human-readable "name(k=v, ...)" dump of the options.
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

  /// @brief Validate option ranges; throws on bad values (kaldi @c Check()).
  /// @see LatticeFasterDecoderConfig
  void Check() const {
    if (!(beam > 0.0 && max_active > 1 && lattice_beam > 0.0 &&
          min_active <= max_active && prune_interval > 0 && beam_delta > 0.0 &&
          hash_ratio >= 1.0 && prune_scale > 0.0 && prune_scale < 1.0)) {
      throw std::runtime_error("LatticeFasterDecoderConfig: invalid options");
    }
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
/// @brief A forward link in the lattice (kaldi's decoder::ForwardLink).
///
/// Links connect tokens of adjacent frames; near-best predecessors within
/// the lattice beam are kept so the raw lattice can be produced at the end.
template <typename Token>
struct ForwardLink {
  using Label = fst::StdArc::Label;

  Token *next_tok;      // the next token [or nullptr if represents final-state]
  Label ilabel;         // ilabel on arc
  Label olabel;         // olabel on arc
  float graph_cost;     // graph cost of traversing arc (contains LM, etc.)
  float acoustic_cost;  // acoustic cost (pre-scaled) of traversing arc
  ForwardLink *next;    // next in singly-linked list of forward arcs (arcs
                        // in the state-level lattice) from a token.
  inline ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                     float graph_cost, float acoustic_cost, ForwardLink *next)
      : next_tok(next_tok),
        ilabel(ilabel),
        olabel(olabel),
        graph_cost(graph_cost),
        acoustic_cost(acoustic_cost),
        next(next) {}
};

/// @brief The standard token: best cost + one backpointer chain
/// (kaldi's decoder::StdToken).
struct StdToken {
  using ForwardLinkT = ForwardLink<StdToken>;
  using Token = StdToken;

  // Standard token type for LatticeFasterDecoder.  Each active HCLG
  // (decoding-graph) state on each frame has one token.

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  float tot_cost;

  // exta_cost is >= 0.  After calling PruneForwardLinks, this equals the
  // minimum difference between the cost of the best path that this link is a
  // part of, and the cost of the absolute best path, under the assumption that
  // any of the currently active states at the decoding front may eventually
  // succeed (e.g. if you were to take the currently active states one by one
  // and compute this difference, and then take the minimum).
  float extra_cost;

  // 'links' is the head of singly-linked list of ForwardLinks, which is what we
  // use for lattice generation.
  ForwardLinkT *links;

  // 'next' is the next in the singly-linked list of tokens for this frame.
  Token *next;

  // This function does nothing and should be optimized out; it's needed
  // so we can share the regular LatticeFasterDecoderTpl code and the code
  // for LatticeFasterOnlineDecoder that supports fast traceback.
  inline void SetBackpointer(Token *backpointer) {}

  // This constructor just ignores the 'backpointer' argument.  That argument is
  // needed so that we can use the same decoder code for LatticeFasterDecoderTpl
  // and LatticeFasterOnlineDecoderTpl (which needs backpointers to support a
  // fast way to obtain the best path).
  inline StdToken(float tot_cost, float extra_cost, ForwardLinkT *links,
                  Token *next, Token *backpointer)
      : tot_cost(tot_cost), extra_cost(extra_cost), links(links), next(next) {}
};

/// @brief Token keeping ALL backpointer links (kaldi's
/// decoder::BackpointerToken).
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

  // 'next' is the next in the singly-linked list of tokens for this frame.
  BackpointerToken *next;

  // Best preceding BackpointerToken (could be a on this frame, connected to
  // this via an epsilon transition, or on a previous frame).  This is only
  // required for an efficient GetBestPath function in
  // LatticeFasterOnlineDecoderTpl; it plays no part in the lattice generation
  // (the "links" list is what stores the forward links, for that).
  Token *backpointer;

  inline void SetBackpointer(Token *backpointer) {
    this->backpointer = backpointer;
  }

  inline BackpointerToken(float tot_cost, float extra_cost, ForwardLinkT *links,
                          Token *next, Token *backpointer)
      : tot_cost(tot_cost),
        extra_cost(extra_cost),
        links(links),
        next(next),
        backpointer(backpointer) {}
};

}  // namespace decoder

/** This is the "normal" lattice-generating decoder.
    See kaldi's doc/lattices.dox and doc/decoders.dox for more information.

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
template <typename FST, typename Token = decoder::StdToken>
class LatticeFasterDecoderTpl {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using ForwardLinkT = decoder::ForwardLink<Token>;

  LatticeFasterDecoderTpl(const LatticeFasterDecoderTpl &) = delete;
  LatticeFasterDecoderTpl &operator=(const LatticeFasterDecoderTpl &) = delete;

  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of
  // 'fst'.
  /// @brief Construct the decoder (graph NOT owned).
  /// @param [in] fst     Decoding graph; must outlive the decoder.
  /// @param [in] config  Beams and limits.
  /// @see Decode, GetBestPath, GetRawLattice
  LatticeFasterDecoderTpl(const FST &fst,
                          const LatticeFasterDecoderConfig &config);

  // This version of the constructor takes ownership of the fst, and will delete
  // it when this object is destroyed.
  /// @brief Construct the decoder taking OWNERSHIP of the graph
  /// (deleted in the destructor).
  /// @param [in] config  Beams and limits.
  /// @param [in] fst     Heap-allocated graph; ownership transferred.
  /// @see LatticeFasterDecoderTpl(const FST &, const Config &)
  LatticeFasterDecoderTpl(const LatticeFasterDecoderConfig &config, FST *fst);

  /// @brief Replace the configuration (between utterances).
  /// @param [in] config  New beams and limits.
  void SetOptions(const LatticeFasterDecoderConfig &config) {
    config_ = config;
  }

  /// @return The current configuration.
  const LatticeFasterDecoderConfig &GetOptions() const { return config_; }

  /// Destructor; frees tokens/links (and the graph if owned).
  ~LatticeFasterDecoderTpl();

  /// Decodes until there are no more frames left in the decodable object.
  /// Note: this may block waiting for input if the decodable object blocks.
  /// Returns true if any kind of traceback is available (not necessarily
  /// from a final state).
  /// @brief Decode to the end of the decodable's frames.
  /// @param [in] decodable  Scores; must outlive the decode.
  /// @return True if any traceback is available.
  /// @see InitDecoding, AdvanceDecoding, GetBestPath
  bool Decode(DecodableInterface *decodable);

  /// says whether a final-state was active on the last frame.  If it was not,
  /// the lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<float>::infinity();
  }

  /// Outputs an FST corresponding to the single best path through the lattice.
  /// Returns true if result is nonempty (using the return status is deprecated,
  /// it will become void).  If "use_final_probs" is true AND we reached the
  /// final-state of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.  Note: this just calls
  /// GetRawLattice() and figures out the shortest path.
  /// @brief 1-best traceback as a linear Lattice FST.
  ///
  /// This is the DECODE-TIME best path -- what kaldi writes to its words and
  /// alignment tables in gmm-latgen-faster.
  /// @param [out] ofst            The best path.
  /// @param [in]  use_final_probs Account for final-probs (default true).
  /// @return True if the output lattice was nonempty.
  /// @see GetRawLattice, ReachedFinal
  bool GetBestPath(fst::Lattice *ofst, bool use_final_probs = true) const;

  /// Outputs an FST corresponding to the raw, state-level
  /// tracebacks.  Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state
  /// of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  /// The raw lattice will be topologically sorted.
  ///
  /// See also GetRawLatticePruned in lattice-faster-online-decoder.h,
  /// which also supports a pruning beam, in case for some reason
  /// you want it pruned tighter than the regular lattice beam.
  /// We could put that here in future needed.
  /// @brief The raw, state-level traceback lattice.
  ///
  /// Alternatives retained within @c lattice_beam; graph costs in Value1,
  /// acoustic costs in Value2.  Topologically sorted.
  /// @param [out] ofst            The raw lattice.
  /// @param [in]  use_final_probs Account for final-probs (default true).
  /// @return True if the output lattice was nonempty.
  /// @see GetBestPath
  bool GetRawLattice(fst::Lattice *ofst, bool use_final_probs = true) const;

  /// [Deprecated, users should now use GetRawLattice and determinize it
  /// themselves, e.g. using DeterminizeLatticePhonePrunedWrapper].
  /// Outputs an FST corresponding to the lattice-determinized
  /// lattice (one path per word sequence).   Returns true if result is
  /// nonempty. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
#if 0
  bool GetLattice(CompactLattice *ofst, bool use_final_probs = true) const;
#endif

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need to
  /// call this.  You can also call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  /// @brief Reset for a new utterance.
  /// @see AdvanceDecoding, FinalizeDecoding, Decode
  void InitDecoding();

  /// This will decode until there are no more frames ready in the decodable
  /// object.  You can keep calling it each time more frames become available.
  /// If max_num_frames is specified, it specifies the maximum number of frames
  /// the function will decode before returning.
  /// @brief Decode up to @c max_num_frames more frames.
  /// @param [in] decodable       Scores; must outlive the decode.
  /// @param [in] max_num_frames  Frame limit, or -1 for no limit.
  /// @see InitDecoding, NumFramesDecoded
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32_t max_num_frames = -1);

  /// This function may be optionally called after AdvanceDecoding(), when you
  /// do not plan to decode any further.  It does an extra pruning step that
  /// will help to prune the lattices output by GetLattice and (particularly)
  /// GetRawLattice more completely, particularly toward the end of the
  /// utterance.  If you call this, you cannot call AdvanceDecoding again (it
  /// will fail), and you cannot call GetLattice() and related functions with
  /// use_final_probs = false.  Used to be called PruneActiveTokensFinal().
  /// @brief Final pruning of forward links; call before GetRawLattice for
  /// the most accurate lattices near the end of the utterance.
  /// @see GetRawLattice
  void FinalizeDecoding();

  /// FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost
  /// plus cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were
  /// present on the final frame.  It will usually be nonnegative.  If it not
  /// too positive (e.g. < 5 is my first guess, but this is not tested) you can
  /// take it as a good indication that we reached the final-state with
  /// reasonable likelihood.
  /// @brief How far the best final cost is from the best cost overall
  /// (infinity means no final states on the last frame).
  /// @return The relative final cost.
  /// @see ReachedFinal
  float FinalRelativeCost() const;

  // Returns the number of frames decoded so far.  The value returned changes
  // whenever we call ProcessEmitting().
  inline int32_t NumFramesDecoded() const { return active_toks_.size() - 1; }

 protected:
  // we make things protected instead of private, as code in
  // LatticeFasterOnlineDecoderTpl, which inherits from this, also uses the
  // internals.

  // Deletes the elements of the singly linked list tok->links.
  void DeleteForwardLinks(Token *tok);

  // head of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks = nullptr;
    bool must_prune_forward_links = true;
    bool must_prune_tokens = true;
    TokenList() = default;
  };

  using Elem = typename HashList<StateId, Token *>::Elem;
  // Equivalent to:
  //  struct Elem {
  //    StateId key;
  //    Token *val;
  //    Elem *tail;
  //  };

  void PossiblyResizeHash(size_t num_toks);

  // FindOrAddToken either locates a token in hash of toks_, or if necessary
  // inserts a new, empty token (i.e. with no forward links) for the current
  // frame.  [note: it's inserted if necessary into hash toks_ and also into the
  // singly linked list of tokens active on this frame (whose head is at
  // active_toks_[frame]).  The frame_plus_one argument is the acoustic frame
  // index plus one, which is used to index into the active_toks_ array.
  // Returns the Token pointer.  Sets "changed" (if non-nullptr) to true if the
  // token was newly created or the cost changed.
  // If Token == StdToken, the 'backpointer' argument has no purpose (and will
  // hopefully be optimized out).
  inline Elem *FindOrAddToken(StateId state, int32_t frame_plus_one,
                              float tot_cost, Token *backpointer,
                              bool *changed);

  // prunes outgoing links for all tokens in active_toks_[frame]
  // it's called by PruneActiveTokens
  // all links, that have link_extra_cost > lattice_beam are pruned
  // delta is the amount by which the extra_costs must change
  // before we set *extra_costs_changed = true.
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned
  void PruneForwardLinks(int32_t frame_plus_one, bool *extra_costs_changed,
                         bool *links_pruned, float delta);

  // This function computes the final-costs for tokens active on the final
  // frame.  It outputs to final-costs, if non-nullptr, a map from the Token*
  // pointer to the final-prob of the corresponding state, for all Tokens
  // that correspond to states that have final-probs.  This map will be
  // empty if there were no final-probs.  It outputs to
  // final_relative_cost, if non-nullptr, the difference between the best
  // forward-cost including the final-prob cost, and the best forward-cost
  // without including the final-prob cost (this will usually be positive), or
  // infinity if there were no final-probs.  [c.f. FinalRelativeCost(), which
  // outputs this quanitity].  It outputs to final_best_cost, if
  // non-nullptr, the lowest for any token t active on the final frame, of
  // forward-cost[t] + final-cost[t], where final-cost[t] is the final-cost in
  // the graph of the state corresponding to token t, or the best of
  // forward-cost[t] if there were no final-probs active on the final frame.
  // You cannot call this after FinalizeDecoding() has been called; in that
  // case you should get the answer from class-member variables.
  void ComputeFinalCosts(std::unordered_map<Token *, float> *final_costs,
                         float *final_relative_cost,
                         float *final_best_cost) const;

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal();

  // Prune away any tokens on this frame that have no forward links.
  // [we don't do this in PruneForwardLinks because it would give us
  // a problem with dangling pointers].
  // It's called by PruneActiveTokens if any forward links have been pruned
  void PruneTokensForFrame(int32_t frame_plus_one);

  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.  It's
  // possible to prove that this is "correct" in the sense that we won't lose
  // anything outside of lat_beam, regardless of what happens in the future.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse
  // less far.
  void PruneActiveTokens(float delta);

  /// Gets the weight cutoff.  Also counts the active tokens.
  float GetCutoff(Elem *list_head, size_t *tok_count, float *adaptive_beam,
                  Elem **best_elem);

  /// Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  /// cur_toks_.  Returns the cost cutoff for subsequent ProcessNonemitting() to
  /// use.
  float ProcessEmitting(DecodableInterface *decodable);

  /// Processes nonemitting (epsilon) arcs for one frame.  Called after
  /// ProcessEmitting() on each frame.  The cost cutoff is computed by the
  /// preceding ProcessEmitting().
  void ProcessNonemitting(float cost_cutoff);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.  It is indexed by frame-index
  // plus one, where the frame-index is zero-based, as used in decodable object.
  // That is, the emitting probs of frame t are accounted for in tokens at
  // toks_[t+1].  The zeroth frame is for nonemitting transition at the start of
  // the graph.
  /// Active tokens of the current frame, indexed by FST state.
  HashList<StateId, Token *> toks_;

  /// Tokens of every frame decoded so far (for traceback).
  std::vector<TokenList> active_toks_;  // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<const Elem *>
      queue_;                     // temp variable used in ProcessNonemitting,
  std::vector<float> tmp_array_;  // used in GetCutoff.

  // fst_ is a pointer to the FST we are decoding from.
  /// The decoding graph (borrowed or owned -- see the constructors).
  const FST *fst_;
  // delete_fst_ is true if the pointer fst_ needs to be deleted when this
  // object is destroyed.
  /// Whether the destructor deletes @c fst_ (owning constructor).
  bool delete_fst_;

  std::vector<float> cost_offsets_;  // This contains, for each
  // frame, an offset that was added to the acoustic log-likelihoods on that
  // frame in order to keep everything in a nice dynamic range i.e.  close to
  // zero, to reduce roundoff errors.
  /// The configuration (beams and limits).
  LatticeFasterDecoderConfig config_;
  /// Number of tokens currently allocated (for hash sizing).
  int32_t num_toks_ = 0;  // current total #toks allocated...
  /// Latch so the "no tokens alive" warning prints once per utterance.
  bool warned_ = false;

  /// decoding_finalized_ is true if someone called FinalizeDecoding().  [note,
  /// calling this is optional].  If true, it's forbidden to decode more.  Also,
  /// if this is set, then the output of ComputeFinalCosts() is in the next
  /// three variables.  The reason we need to do this is that after
  /// FinalizeDecoding() calls PruneTokensForFrame() for the final frame, some
  /// of the tokens on the last frame are freed, so we free the list from toks_
  /// to avoid having dangling pointers hanging around.
  /// Whether FinalizeDecoding() has run for the current utterance.
  bool decoding_finalized_ = false;
  /// For the meaning of the next 3 variables, see the comment for
  /// decoding_finalized_ above., and ComputeFinalCosts().
  std::unordered_map<Token *, float> final_costs_;
  float final_relative_cost_ = 0.0f;
  float final_best_cost_ = 0.0f;

  // Memory pools for storing tokens and forward links.
  // We use it to decrease the work put on allocator and to move some of data
  // together. Too small block sizes will result in more work to allocator but
  // bigger ones increase the memory usage.
  fst::MemoryPool<Token> token_pool_;
  fst::MemoryPool<ForwardLinkT> forward_link_pool_;

  // There are various cleanup tasks... the toks_ structure contains
  // singly linked lists of Token pointers, where Elem is the list type.
  // It also indexes them in a hash, indexed by state (this hash is only
  // maintained for the most recent frame).  toks_.Clear()
  // deletes them from the hash and returns the list of Elems.  The
  // function DeleteElems calls toks_.Delete(elem) for each elem in
  // the list, which returns ownership of the Elem to the toks_ structure
  // for reuse, but does not delete the Token pointer.  The Token pointers
  // are reference-counted and are ultimately deleted in PruneTokensForFrame,
  // but are also linked together on each frame by their own linked-list,
  // using the "next" pointer.  We delete them manually.
  void DeleteElems(Elem *list);

  // This function takes a singly linked list of tokens for a single frame, and
  // outputs a list of them in topological order (it will crash if no such order
  // can be found, which will typically be due to decoding graphs with epsilon
  // cycles, which are not allowed).  Note: the output list may contain NULLs,
  // which the caller should pass over; it just happens to be more efficient for
  // the algorithm to output a list that contains NULLs.
  static void TopSortTokens(Token *tok_list,
                            std::vector<Token *> *topsorted_list);

  void ClearActiveTokens();
};

typedef LatticeFasterDecoderTpl<fst::StdFst, decoder::StdToken>
    LatticeFasterDecoder;

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_LATTICE_FASTER_DECODER_H_
