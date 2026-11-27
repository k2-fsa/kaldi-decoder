// kaldi-decoder/csrc/lattice-faster-online-decoder.cc

// Copyright 2009-2012  Microsoft Corporation  Mirko Hannemann
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2014  IMSL, PKU-HKUST (author: Wei Shi)
//                2018  Zhehuai Chen
// Copyright (c)  2026  Xiaomi Corporation

// This file is copied/modified from
// kaldi/src/decoder/lattice-faster-online-decoder.cc

#include "kaldi-decoder/csrc/lattice-faster-online-decoder.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kaldi-decoder/csrc/macros.h"

namespace kaldi_decoder {

template <typename FST>
bool LatticeFasterOnlineDecoderTpl<FST>::GetBestPath(
    fst::Lattice *olat, bool use_final_probs) const {
  olat->DeleteStates();
  float final_graph_cost = 0;
  BestPathIterator iter = BestPathEnd(use_final_probs, &final_graph_cost);
  if (iter.Done()) return false;  // would have printed warning.
  StateId state = olat->AddState();
  olat->SetFinal(state, fst::LatticeWeight(final_graph_cost, 0.0));
  while (!iter.Done()) {
    fst::LatticeArc arc;
    iter = TraceBackBestPath(iter, &arc);
    arc.nextstate = state;
    StateId new_state = olat->AddState();
    olat->AddArc(new_state, arc);
    state = new_state;
  }
  olat->SetStart(state);
  return true;
}

template <typename FST>
typename LatticeFasterOnlineDecoderTpl<FST>::BestPathIterator
LatticeFasterOnlineDecoderTpl<FST>::BestPathEnd(bool use_final_probs,
                                                float *final_cost_out) const {
  if (this->decoding_finalized_ && !use_final_probs) {
    throw std::runtime_error(
        "You cannot call FinalizeDecoding() and then call "
        "BestPathEnd() with use_final_probs == false");
  }
  if (this->NumFramesDecoded() <= 0) {
    throw std::runtime_error(
        "You cannot call BestPathEnd if no frames were decoded.");
  }

  std::unordered_map<Token *, float> final_costs_local;

  const std::unordered_map<Token *, float> &final_costs =
      (this->decoding_finalized_ ? this->final_costs_ : final_costs_local);
  if (!this->decoding_finalized_ && use_final_probs)
    this->ComputeFinalCosts(&final_costs_local, nullptr, nullptr);

  // Singly linked list of tokens on last frame (access list through "next"
  // pointer).
  float best_cost = std::numeric_limits<float>::infinity();
  float best_final_cost = 0;
  Token *best_tok = nullptr;
  for (Token *tok = this->active_toks_.back().toks; tok != nullptr;
       tok = tok->next) {
    float cost = tok->tot_cost;
    float final_cost = 0.0;
    if (use_final_probs && !final_costs.empty()) {
      // if we are instructed to use final-probs, and any final tokens were
      // active on final frame, include the final-prob in the cost of the
      // token.
      auto iter = final_costs.find(tok);
      if (iter != final_costs.end()) {
        final_cost = iter->second;
        cost += final_cost;
      } else {
        cost = std::numeric_limits<float>::infinity();
      }
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_tok = tok;
      best_final_cost = final_cost;
    }
  }
  if (best_tok == nullptr) {
    // this should not happen, and is likely a code error or caused by
    // infinities in likelihoods, but I'm not making it a fatal error for now.
    KALDI_DECODER_LOGE("No final token found.");
  }
  if (final_cost_out) *final_cost_out = best_final_cost;
  return BestPathIterator(best_tok, this->NumFramesDecoded() - 1);
}

template <typename FST>
typename LatticeFasterOnlineDecoderTpl<FST>::BestPathIterator
LatticeFasterOnlineDecoderTpl<FST>::TraceBackBestPath(
    BestPathIterator iter, fst::LatticeArc *oarc) const {
  if (iter.Done() || oarc == nullptr) {
    throw std::runtime_error(
        "TraceBackBestPath: iterator is done or arc is null");
  }
  Token *tok = static_cast<Token *>(iter.tok);
  int32_t cur_t = iter.frame;
  int32_t step_t = 0;
  if (tok->backpointer != nullptr) {
    // retrieve the correct forward link (with the best link cost)
    float best_cost = std::numeric_limits<float>::infinity();
    ForwardLinkT *link = nullptr;
    for (link = tok->backpointer->links; link != nullptr; link = link->next) {
      if (link->next_tok == tok) {  // this is a link to "tok"
        float graph_cost = link->graph_cost;
        float acoustic_cost = link->acoustic_cost;
        float cost = graph_cost + acoustic_cost;
        if (cost < best_cost) {
          oarc->ilabel = link->ilabel;
          oarc->olabel = link->olabel;
          if (link->ilabel != 0) {
            if (static_cast<size_t>(cur_t) >= this->cost_offsets_.size()) {
              throw std::runtime_error(
                  "TraceBackBestPath: frame index out of range");
            }
            acoustic_cost -= this->cost_offsets_[cur_t];
            step_t = -1;
          } else {
            step_t = 0;
          }
          oarc->weight = fst::LatticeWeight(graph_cost, acoustic_cost);
          best_cost = cost;
        }
      }
    }
    if (link == nullptr &&
        best_cost == std::numeric_limits<float>::infinity()) {
      // Did not find correct link.
      throw std::runtime_error(
          "Error tracing best-path back (likely bug in token-pruning "
          "algorithm)");
    }
  } else {
    oarc->ilabel = 0;
    oarc->olabel = 0;
    oarc->weight = fst::LatticeWeight::One();  // zero costs.
  }
  return BestPathIterator(tok->backpointer, cur_t + step_t);
}

template <typename FST>
bool LatticeFasterOnlineDecoderTpl<FST>::GetRawLatticePruned(
    fst::Lattice *ofst, bool use_final_probs, float beam) const {
  using LArc = fst::LatticeArc;
  using LStateId = LArc::StateId;
  using LWeight = LArc::Weight;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (this->decoding_finalized_ && !use_final_probs) {
    throw std::runtime_error(
        "You cannot call FinalizeDecoding() and then call "
        "GetRawLattice() with use_final_probs == false");
  }

  std::unordered_map<Token *, float> final_costs_local;

  const std::unordered_map<Token *, float> &final_costs =
      (this->decoding_finalized_ ? this->final_costs_ : final_costs_local);
  if (!this->decoding_finalized_ && use_final_probs)
    this->ComputeFinalCosts(&final_costs_local, nullptr, nullptr);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32_t num_frames = static_cast<int32_t>(this->active_toks_.size()) - 1;
  if (num_frames <= 0) {
    throw std::runtime_error("GetRawLatticePruned: no frames decoded");
  }
  for (int32_t f = 0; f <= num_frames; f++) {
    if (this->active_toks_[f].toks == nullptr) {
      KALDI_DECODER_LOGE("No tokens active on frame %d: not producing lattice.",
                         f);
      return false;
    }
  }
  std::unordered_map<Token *, LStateId> tok_map;
  std::queue<std::pair<Token *, int32_t>> tok_queue;
  // First initialize the queue and states.  Put the initial state on the
  // queue; this is the last token in the list active_toks_[0].toks.
  for (Token *tok = this->active_toks_[0].toks; tok != nullptr;
       tok = tok->next) {
    if (tok->next == nullptr) {
      tok_map[tok] = ofst->AddState();
      ofst->SetStart(tok_map[tok]);
      std::pair<Token *, int32_t> tok_pair(tok, 0);  // #frame = 0
      tok_queue.push(tok_pair);
    }
  }

  // Next create states for "good" tokens
  while (!tok_queue.empty()) {
    std::pair<Token *, int32_t> cur_tok_pair = tok_queue.front();
    tok_queue.pop();
    Token *cur_tok = cur_tok_pair.first;
    int32_t cur_frame = cur_tok_pair.second;
    if (cur_frame < 0 ||
        cur_frame > static_cast<int32_t>(this->cost_offsets_.size())) {
      throw std::runtime_error("GetRawLatticePruned: bad frame index");
    }

    auto iter = tok_map.find(cur_tok);
    if (iter == tok_map.end()) {
      throw std::runtime_error("GetRawLatticePruned: token not in map");
    }
    LStateId cur_state = iter->second;

    for (ForwardLinkT *l = cur_tok->links; l != nullptr; l = l->next) {
      Token *next_tok = l->next_tok;
      if (next_tok->extra_cost < beam) {
        // so both the current and the next token are good; create the arc
        int32_t next_frame = l->ilabel == 0 ? cur_frame : cur_frame + 1;
        LStateId nextstate = 0;
        if (tok_map.find(next_tok) == tok_map.end()) {
          nextstate = tok_map[next_tok] = ofst->AddState();
          tok_queue.push(std::pair<Token *, int32_t>(next_tok, next_frame));
        } else {
          nextstate = tok_map[next_tok];
        }
        float cost_offset =
            (l->ilabel != 0 ? this->cost_offsets_[cur_frame] : 0);
        LArc arc(l->ilabel, l->olabel,
                 LWeight(l->graph_cost, l->acoustic_cost - cost_offset),
                 nextstate);
        ofst->AddArc(cur_state, arc);
      }
    }
    if (cur_frame == num_frames) {
      if (use_final_probs && !final_costs.empty()) {
        auto fiter = final_costs.find(cur_tok);
        if (fiter != final_costs.end())
          ofst->SetFinal(cur_state, fst::LatticeWeight(fiter->second, 0));
      } else {
        ofst->SetFinal(cur_state, fst::LatticeWeight::One());
      }
    }
  }
  return (ofst->NumStates() != 0);
}

template <typename FST>
std::vector<int32_t> GetBestPathTransitionIds(
    const LatticeFasterOnlineDecoderTpl<FST> &decoder, bool use_final_probs) {
  using Decoder = LatticeFasterOnlineDecoderTpl<FST>;
  int32_t num_frames = decoder.NumFramesDecoded();
  std::vector<int32_t> ans;
  if (num_frames <= 0) return ans;
  typename Decoder::BestPathIterator iter =
      decoder.BestPathEnd(use_final_probs, nullptr);
  if (iter.Done()) return ans;
  ans.resize(num_frames);
  // Same loop as kaldi's OnlineSilenceWeighting::ComputeCurrentTraceback().
  int32_t frame = num_frames - 1;
  while (frame >= 0) {
    fst::LatticeArc arc;
    arc.ilabel = 0;
    while (arc.ilabel == 0)  // the while loop skips over input-epsilons
      iter = decoder.TraceBackBestPath(iter, &arc);
    // note, the iter.frame values are slightly unintuitively defined,
    // they are one less than you might expect.
    if (iter.frame != frame - 1) {
      throw std::runtime_error(
          "GetBestPathTransitionIds: traceback frame mismatch");
    }
    ans[frame] = arc.ilabel;
    frame--;
  }
  return ans;
}

// Instantiate the template for the FST types that we'll need.
template class LatticeFasterOnlineDecoderTpl<fst::Fst<fst::StdArc>>;
template class LatticeFasterOnlineDecoderTpl<fst::VectorFst<fst::StdArc>>;
template class LatticeFasterOnlineDecoderTpl<fst::ConstFst<fst::StdArc>>;

template std::vector<int32_t> GetBestPathTransitionIds(
    const LatticeFasterOnlineDecoderTpl<fst::Fst<fst::StdArc>> &decoder,
    bool use_final_probs);

}  // namespace kaldi_decoder
