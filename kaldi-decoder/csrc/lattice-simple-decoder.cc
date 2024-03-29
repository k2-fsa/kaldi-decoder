// kaldi-decoder/csrc/lattice-simple-decoder.cc

// Copyright 2009-2012  Microsoft Corporation
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/decoder/lattice-simple-decoder.cc

#include "kaldi-decoder/csrc/lattice-simple-decoder.h"

#include "kaldi-decoder/csrc/kaldi-math.h"

namespace kaldi_decoder {

void LatticeSimpleDecoder::InitDecoding() {
  // clean up from last time:
  cur_toks_.clear();
  prev_toks_.clear();
  ClearActiveTokens();
  warned_ = false;
  decoding_finalized_ = false;
  final_costs_.clear();
  num_toks_ = 0;
  StateId start_state = fst_.Start();
  KALDI_DECODER_ASSERT(start_state != fst::kNoStateId);
  active_toks_.resize(1);
  Token *start_tok = new Token(0.0, 0.0, nullptr, nullptr);
  active_toks_[0].toks = start_tok;
  cur_toks_[start_state] = start_tok;
  num_toks_++;
  ProcessNonemitting();
}

void LatticeSimpleDecoder::ClearActiveTokens() {  // a cleanup routine, at utt
                                                  // end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != nullptr;) {
      tok->DeleteForwardLinks();
      Token *next_tok = tok->next;
      delete tok;
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_DECODER_ASSERT(num_toks_ == 0);
}

bool LatticeSimpleDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();

  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }

    ProcessEmitting(decodable);
    // Important to call PruneCurrentTokens before ProcessNonemitting, or we
    // would get dangling forward pointers.  Anyway, ProcessNonemitting uses the
    // beam.
    PruneCurrentTokens(config_.beam, &cur_toks_);
    ProcessNonemitting();
  }
  FinalizeDecoding();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !final_costs_.empty();
}

// FindOrAddToken either locates a token in cur_toks_, or if necessary inserts a
// new, empty token (i.e. with no forward links) for the current frame.  [note:
// it's inserted if necessary into cur_toks_ and also into the singly linked
// list of tokens active on this frame (whose head is at active_toks_[frame]).
//
// Returns the Token pointer.  Sets "changed" (if non-NULL) to true
// if the token was newly created or the cost changed.
LatticeSimpleDecoder::Token *LatticeSimpleDecoder::FindOrAddToken(
    StateId state, int32_t frame, float tot_cost, bool emitting,
    bool *changed) {
  KALDI_DECODER_ASSERT(frame < active_toks_.size());
  Token *&toks = active_toks_[frame].toks;

  auto find_iter = cur_toks_.find(state);
  if (find_iter == cur_toks_.end()) {  // no such token presently.
    // Create one.
    const float extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up
    // on the winning path.
    Token *new_tok = new Token(tot_cost, extra_cost, nullptr, toks);
    toks = new_tok;
    num_toks_++;
    cur_toks_[state] = new_tok;

    if (changed) {
      *changed = true;
    }

    return new_tok;
  } else {
    Token *tok =
        find_iter->second;  // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) {
      tok->tot_cost = tot_cost;
      if (changed) {
        *changed = true;
      }
    } else {
      if (changed) {
        *changed = false;
      }
    }
    return tok;
  }
}

void LatticeSimpleDecoder::ProcessNonemitting() {
  KALDI_DECODER_ASSERT(!active_toks_.empty());
  int32_t frame = static_cast<int32_t>(active_toks_.size()) - 2;
  // Note: "frame" is the time-index we just processed, or -1 if
  // we are processing the nonemitting transitions before the
  // first frame (called from InitDecoding()).

  // Processes nonemitting arcs for one frame.  Propagates within
  // cur_toks_.  Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.
  std::vector<StateId> queue;
  float best_cost = std::numeric_limits<float>::infinity();
  for (auto iter = cur_toks_.begin(); iter != cur_toks_.end(); ++iter) {
    StateId state = iter->first;

    if (fst_.NumInputEpsilons(state) != 0) {
      queue.push_back(state);
    }

    best_cost = std::min(best_cost, iter->second->tot_cost);
  }

  if (queue.empty()) {
    if (!warned_) {
      KALDI_DECODER_LOG
          << "Error in ProcessNonEmitting: no surviving tokens: frame is "
          << frame;
      warned_ = true;
    }
  }
  float cutoff = best_cost + config_.beam;

  while (!queue.empty()) {
    StateId state = queue.back();
    queue.pop_back();
    Token *tok = cur_toks_[state];
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    tok->DeleteForwardLinks();
    tok->links = nullptr;
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        float graph_cost = arc.weight.Value();
        float cur_cost = tok->tot_cost;
        float tot_cost = cur_cost + graph_cost;

        if (tot_cost < cutoff) {
          bool changed;
          Token *new_tok = FindOrAddToken(arc.nextstate, frame + 1, tot_cost,
                                          false, &changed);

          tok->links = new ForwardLink(new_tok, 0, arc.olabel, graph_cost, 0,
                                       tok->links);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          if (changed && fst_.NumInputEpsilons(arc.nextstate) != 0) {
            queue.push_back(arc.nextstate);
          }
        }
      }
    }
  }
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame
// before that.  We go backwards through the frames and stop when we reach a
// point where the delta-costs are not changing (and the delta controls when we
// consider a cost to have "not changed").
void LatticeSimpleDecoder::PruneActiveTokens(float delta) {
  int32_t cur_frame_plus_one = NumFramesDecoded();
  int32_t num_toks_begin = num_toks_;
  // The index "f" below represents a "frame plus one", i.e. you'd have to
  // subtract one to get the corresponding index for the decodable object.
  for (int32_t f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them
    // (2) we never pruned the forward links on the next frame, which
    //
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && f > 0)
        active_toks_[f - 1].must_prune_forward_links = true;
      if (links_pruned) active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false;
    }
    if (f + 1 < cur_frame_plus_one && active_toks_[f + 1].must_prune_tokens) {
      PruneTokensForFrame(f + 1);
      active_toks_[f + 1].must_prune_tokens = false;
    }
  }
  KALDI_DECODER_LOG << "PruneActiveTokens: pruned tokens from "
                    << num_toks_begin << " to " << num_toks_;
}

// delta is the amount by which the extra_costs must
// change before it sets "extra_costs_changed" to true.  If delta is larger,
// we'll tend to go back less far toward the beginning of the file.
void LatticeSimpleDecoder::PruneForwardLinks(int32_t frame,
                                             bool *extra_costs_changed,
                                             bool *links_pruned, float delta) {
  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_DECODER_ASSERT(frame >= 0 && frame < active_toks_.size());
  if (active_toks_[frame].toks == nullptr) {  // empty list; this should
    // not happen.
    if (!warned_) {
      KALDI_DECODER_WARN << "No tokens alive [doing pruning].. warning first "
                            "time only for each utterance\n";
      warned_ = true;
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame].toks; tok != nullptr;
         tok = tok->next) {
      ForwardLink *link, *prev_link = nullptr;
      // will recompute tok_extra_cost.
      float tok_extra_cost = std::numeric_limits<float>::infinity();
      for (link = tok->links; link != nullptr;) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        float link_extra_cost =
            next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost) -
             next_tok->tot_cost);
        KALDI_DECODER_ASSERT(link_extra_cost ==
                             link_extra_cost);  // check for NaN

        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != nullptr) {
            prev_link->next = next_link;
          } else {
            tok->links = next_link;
          }

          delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {  // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (link_extra_cost < -0.01) {
              KALDI_DECODER_WARN << "Negative extra_cost: " << link_extra_cost;
            }
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) {
            tok_extra_cost = link_extra_cost;
          }

          prev_link = link;
          link = link->next;
        }
      }
      if (fabs(tok_extra_cost - tok->extra_cost) > delta) {
        changed = true;
      }

      tok->extra_cost =
          tok_extra_cost;  // will be +infinity or <= lattice_beam_.
    }
    if (changed) {
      *extra_costs_changed = true;
    }

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  }
}

// Prune away any tokens on this frame that have no forward links. [we don't do
// this in PruneForwardLinks because it would give us a problem with dangling
// pointers].
void LatticeSimpleDecoder::PruneTokensForFrame(int32_t frame) {
  KALDI_DECODER_ASSERT(frame >= 0 && frame < active_toks_.size());
  Token *&toks = active_toks_[frame].toks;
  if (toks == nullptr) {
    KALDI_DECODER_WARN << "No tokens alive [doing pruning]";
  }

  Token *tok, *next_tok, *prev_tok = nullptr;
  for (tok = toks; tok != nullptr; tok = next_tok) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<float>::infinity()) {
      // Next token is unreachable from end of graph; excise tok from list
      // and delete tok.
      if (prev_tok != nullptr) {
        prev_tok->next = tok->next;
      } else {
        toks = tok->next;
      }
      delete tok;
      num_toks_--;
    } else {
      prev_tok = tok;
    }
  }
}

// PruneCurrentTokens deletes the tokens from the "toks" map, but not
// from the active_toks_ list, which could cause dangling forward pointers
// (will delete it during regular pruning operation).
void LatticeSimpleDecoder::PruneCurrentTokens(
    float beam, std::unordered_map<StateId, Token *> *toks) {
  if (toks->empty()) {
    KALDI_DECODER_LOG << "No tokens to prune.\n";
    return;
  }
  float best_cost = 1.0e+10;  // positive == high cost == bad.
  for (auto iter = toks->begin(); iter != toks->end(); ++iter) {
    best_cost = std::min(best_cost, static_cast<float>(iter->second->tot_cost));
  }
  std::vector<StateId> retained;
  float cutoff = best_cost + beam;
  for (auto iter = toks->begin(); iter != toks->end(); ++iter) {
    if (iter->second->tot_cost < cutoff) {
      retained.push_back(iter->first);
    }
  }
  std::unordered_map<StateId, Token *> tmp;
  for (size_t i = 0; i < retained.size(); i++) {
    tmp[retained[i]] = (*toks)[retained[i]];
  }
  KALDI_DECODER_LOG << "Pruned to " << (retained.size()) << " toks.\n";
  tmp.swap(*toks);
}

void LatticeSimpleDecoder::ProcessEmitting(DecodableInterface *decodable) {
  int32_t frame = static_cast<int32_t>(active_toks_.size()) -
                  1;  // frame is the frame-index
                      // (zero-based) used to get likelihoods
                      // from the decodable object.
  active_toks_.resize(active_toks_.size() + 1);
  prev_toks_.clear();
  cur_toks_.swap(prev_toks_);

  // Processes emitting arcs for one frame.  Propagates from
  // prev_toks_ to cur_toks_.
  float cutoff = std::numeric_limits<float>::infinity();
  for (auto iter = prev_toks_.begin(); iter != prev_toks_.end(); ++iter) {
    StateId state = iter->first;
    Token *tok = iter->second;
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // propagate..
        float ac_cost = -decodable->LogLikelihood(frame, arc.ilabel),
              graph_cost = arc.weight.Value(), cur_cost = tok->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost;
        if (tot_cost >= cutoff) {
          continue;
        } else if (tot_cost + config_.beam < cutoff) {
          cutoff = tot_cost + config_.beam;
        }

        // AddToken adds the next_tok to cur_toks_ (if not already present).
        Token *next_tok =
            FindOrAddToken(arc.nextstate, frame + 1, tot_cost, true, NULL);

        // Add ForwardLink from tok to next_tok (put on head of list tok->links)
        tok->links = new ForwardLink(next_tok, arc.ilabel, arc.olabel,
                                     graph_cost, ac_cost, tok->links);
      }
    }
  }
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
void LatticeSimpleDecoder::FinalizeDecoding() {
  int32_t final_frame_plus_one = NumFramesDecoded();
  int32_t num_toks_begin = num_toks_;
  PruneForwardLinksFinal();
  for (int32_t f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2;  // values not used.
    float dontcare = 0.0;
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_DECODER_LOG << "pruned tokens from " << num_toks_begin << " to "
                    << num_toks_;
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses the
// final-probs for pruning, otherwise it treats all tokens as final.
void LatticeSimpleDecoder::PruneForwardLinksFinal() {
  KALDI_DECODER_ASSERT(!active_toks_.empty());
  int32_t frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks ==
      nullptr) {  // empty list; should not happen.
    KALDI_DECODER_WARN << "No tokens alive at end of file\n";
  }

  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;
  // We're about to delete some of the tokens active on the final frame, so we
  // clear cur_toks_ because otherwise it would then contain dangling pointers.
  cur_toks_.clear();

  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  float delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks; tok != nullptr;
         tok = tok->next) {
      ForwardLink *link, *prev_link = nullptr;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to
      // infinity below we set it to the difference between the
      // (score+final_prob) of this token, and the best such (score+final_prob).

      float final_cost;
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        auto iter = final_costs_.find(tok);
        if (iter != final_costs_.end()) {
          final_cost = iter->second;
        } else {
          final_cost = std::numeric_limits<float>::infinity();
        }
      }
      float tok_extra_cost = tok->tot_cost + final_cost - final_best_cost_;
      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      for (link = tok->links; link != nullptr;) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        float link_extra_cost =
            next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost) -
             next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != nullptr) {
            prev_link->next = next_link;
          } else {
            tok->links = next_link;
          }
          delete link;
          link = next_link;  // advance link but leave prev_link the same.
        } else {  // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (link_extra_cost < -0.01) {
              KALDI_DECODER_WARN << "Negative extra_cost: " << link_extra_cost;
            }

            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) {
            tok_extra_cost = link_extra_cost;
          }

          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_extra_cost > config_.lattice_beam) {
        tok_extra_cost = std::numeric_limits<float>::infinity();
      }
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta)) {
        changed = true;
      }

      // will be +infinity or <= lattice_beam_.
      tok->extra_cost = tok_extra_cost;
    }
  }  // while changed
}

void LatticeSimpleDecoder::ComputeFinalCosts(
    std::unordered_map<Token *, float> *final_costs, float *final_relative_cost,
    float *final_best_cost) const {
  KALDI_DECODER_ASSERT(!decoding_finalized_);
  if (final_costs != nullptr) {
    final_costs->clear();
  }

  float infinity = std::numeric_limits<float>::infinity();
  float best_cost = infinity, best_cost_with_final = infinity;

  for (auto iter = cur_toks_.begin(); iter != cur_toks_.end(); ++iter) {
    StateId state = iter->first;
    Token *tok = iter->second;
    float final_cost = fst_.Final(state).Value();
    float cost = tok->tot_cost, cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    if (final_costs != nullptr && final_cost != infinity) {
      (*final_costs)[tok] = final_cost;
    }
  }
  if (final_relative_cost != nullptr) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != nullptr) {
    if (best_cost_with_final != infinity) {  // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else {  // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}

float LatticeSimpleDecoder::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    float relative_cost;
    ComputeFinalCosts(nullptr, &relative_cost, nullptr);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}

bool LatticeSimpleDecoder::GetBestPath(fst::Lattice *ofst,
                                       bool use_final_probs) const {
  fst::VectorFst<fst::LatticeArc> fst;
  GetRawLattice(&fst, use_final_probs);
  ShortestPath(fst, ofst);
  return (ofst->NumStates() > 0);
}

// Outputs an FST corresponding to the raw, state-level
// tracebacks.
bool LatticeSimpleDecoder::GetRawLattice(fst::Lattice *ofst,
                                         bool use_final_probs) const {
  using Arc = fst::LatticeArc;
  using StateId = Arc::StateId;
  using Weight = Arc::Weight;
  using Label = Arc::Label;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs) {
    KALDI_DECODER_ERR << "You cannot call FinalizeDecoding() and then call "
                      << "GetRawLattice() with use_final_probs == false";
  }

  std::unordered_map<Token *, float> final_costs_local;

  const std::unordered_map<Token *, float> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);

  if (!decoding_finalized_ && use_final_probs) {
    ComputeFinalCosts(&final_costs_local, nullptr, nullptr);
  }

  ofst->DeleteStates();
  int32_t num_frames = NumFramesDecoded();
  KALDI_DECODER_ASSERT(num_frames > 0);
  const int32_t bucket_count = num_toks_ / 2 + 3;
  std::unordered_map<Token *, StateId> tok_map(bucket_count);
  // First create all states.
  for (int32_t f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == nullptr) {
      KALDI_DECODER_WARN << "GetRawLattice: no tokens active on frame " << f
                         << ": not producing lattice.\n";
      return false;
    }
    for (Token *tok = active_toks_[f].toks; tok != nullptr; tok = tok->next) {
      tok_map[tok] = ofst->AddState();
    }
    // The next statement sets the start state of the output FST.
    // Because we always add new states to the head of the list
    // active_toks_[f].toks, and the start state was the first one
    // added, it will be the last one added to ofst.
    if (f == 0 && ofst->NumStates() > 0) {
      ofst->SetStart(ofst->NumStates() - 1);
    }
  }
  StateId cur_state = 0;  // we rely on the fact that we numbered these
  // consecutively (AddState() returns the numbers in order..)
  for (int32_t f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != nullptr;
         tok = tok->next, cur_state++) {
      for (ForwardLink *l = tok->links; l != nullptr; l = l->next) {
        auto iter = tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_DECODER_ASSERT(iter != tok_map.end());
        Arc arc(l->ilabel, l->olabel, Weight(l->graph_cost, l->acoustic_cost),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          auto iter = final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, fst::LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, fst::LatticeWeight::One());
        }
      }
    }
  }
  KALDI_DECODER_ASSERT(cur_state == ofst->NumStates());
  return (cur_state != 0);
}

}  // namespace kaldi_decoder
