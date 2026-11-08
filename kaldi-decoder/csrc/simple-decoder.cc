// kaldi-decoder/csrc/simple-decoder.cc

// Copyright 2009-2011 Microsoft Corporation
//           2012-2013 Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023 Xiaomi Corporation

#include "kaldi-decoder/csrc/simple-decoder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "kaldi-decoder/csrc/macros.h"
#include "kaldifst/csrc/remove-eps-local.h"

namespace kaldi_decoder {

SimpleDecoder::~SimpleDecoder() {
  ClearToks(cur_toks_);
  ClearToks(prev_toks_);
}

bool SimpleDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();
  AdvanceDecoding(decodable);
  return (!cur_toks_.empty());
}

void SimpleDecoder::InitDecoding() {
  // clean up from last time:
  ClearToks(cur_toks_);
  ClearToks(prev_toks_);
  // initialize decoding:
  StateId start_state = fst_.Start();
  if (start_state == fst::kNoStateId) {
    throw std::runtime_error("SimpleDecoder::InitDecoding: FST has no start");
  }
  StdArc dummy_arc(0, 0, StdWeight::One(), start_state);
  cur_toks_[start_state] = new Token(dummy_arc, 0.0, nullptr);
  num_frames_decoded_ = 0;
  ProcessNonemitting();
}

void SimpleDecoder::AdvanceDecoding(DecodableInterface *decodable,
                                    int32_t max_num_frames /*= -1*/) {
  if (num_frames_decoded_ < 0) {
    throw std::runtime_error(
        "SimpleDecoder::AdvanceDecoding: call InitDecoding() first");
  }
  int32_t num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  if (num_frames_ready < num_frames_decoded_) {
    throw std::runtime_error(
        "SimpleDecoder::AdvanceDecoding: num_frames_ready decreased");
  }
  int32_t target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0) {
    target_frames_decoded =
        std::min(target_frames_decoded, num_frames_decoded_ + max_num_frames);
  }

  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    ClearToks(prev_toks_);
    cur_toks_.swap(prev_toks_);
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &cur_toks_);
  }
}

bool SimpleDecoder::ReachedFinal() const {
  for (const auto &kv : cur_toks_) {
    if (kv.second->cost_ != std::numeric_limits<float>::infinity() &&
        fst_.Final(kv.first) != StdWeight::Zero()) {
      return true;
    }
  }
  return false;
}

float SimpleDecoder::FinalRelativeCost() const {
  // as a special case, if there are no active tokens at all (e.g. some kind of
  // pruning failure), return infinity.
  double infinity = std::numeric_limits<double>::infinity();
  if (cur_toks_.empty()) return infinity;
  double best_cost = infinity;
  double best_cost_with_final = infinity;
  for (const auto &kv : cur_toks_) {
    // Note: Plus is taking the minimum cost, since we're in the tropical
    // semiring.
    best_cost = std::min(best_cost, kv.second->cost_);
    best_cost_with_final = std::min(
        best_cost_with_final, kv.second->cost_ + fst_.Final(kv.first).Value());
  }
  float extra_cost = best_cost_with_final - best_cost;
  if (extra_cost != extra_cost) {  // NaN.  This shouldn't happen; it indicates
                                   // some kind of error, most likely.
    KALDI_DECODER_LOGE("Found NaN (likely search failure in decoding)");
    return infinity;
  }
  // Note: extra_cost will be infinity if no states were final.
  return extra_cost;
}

// Outputs an FST corresponding to the single best path
// through the lattice.
bool SimpleDecoder::GetBestPath(fst::Lattice *fst_out,
                                bool use_final_probs) const {
  fst_out->DeleteStates();
  Token *best_tok = nullptr;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (const auto &kv : cur_toks_) {
      if (best_tok == nullptr || *best_tok < *(kv.second)) {
        best_tok = kv.second;
      }
    }
  } else {
    double infinity = std::numeric_limits<double>::infinity(),
           best_cost = infinity;
    for (const auto &kv : cur_toks_) {
      double this_cost = kv.second->cost_ + fst_.Final(kv.first).Value();
      if (this_cost != infinity && this_cost < best_cost) {
        best_cost = this_cost;
        best_tok = kv.second;
      }
    }
  }
  if (best_tok == nullptr) return false;  // No output.

  std::vector<fst::LatticeArc> arcs_reverse;  // arcs in reverse order.
  for (Token *tok = best_tok; tok != nullptr; tok = tok->prev_)
    arcs_reverse.push_back(tok->arc_);
  if (arcs_reverse.back().nextstate != fst_.Start()) {
    throw std::runtime_error("SimpleDecoder::GetBestPath: bad traceback");
  }
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size()) - 1; i >= 0; i--) {
    fst::LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs)
    fst_out->SetFinal(
        cur_state,
        fst::LatticeWeight(fst_.Final(best_tok->arc_.nextstate).Value(), 0.0));
  else
    fst_out->SetFinal(cur_state, fst::LatticeWeight::One());
  fst::RemoveEpsLocal(fst_out);
  return true;
}

void SimpleDecoder::ProcessEmitting(DecodableInterface *decodable) {
  int32_t frame = num_frames_decoded_;
  // Processes emitting arcs for one frame.  Propagates from
  // prev_toks_ to cur_toks_.
  double cutoff = std::numeric_limits<float>::infinity();
  for (const auto &kv : prev_toks_) {
    StateId state = kv.first;
    Token *tok = kv.second;
    if (state != tok->arc_.nextstate) {
      throw std::runtime_error("SimpleDecoder::ProcessEmitting: bad token");
    }
    for (fst::ArcIterator<fst::Fst<StdArc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const StdArc &arc = aiter.Value();
      if (arc.ilabel == 0) {
        continue;
      }

      // propagate..
      float acoustic_cost = -decodable->LogLikelihood(frame, arc.ilabel);
      double total_cost = tok->cost_ + arc.weight.Value() + acoustic_cost;

      if (total_cost >= cutoff) {
        continue;
      }

      if (total_cost + beam_ < cutoff) {
        cutoff = total_cost + beam_;
      }

      Token *new_tok = new Token(arc, acoustic_cost, tok);
      auto find_iter = cur_toks_.find(arc.nextstate);
      if (find_iter == cur_toks_.end()) {
        cur_toks_[arc.nextstate] = new_tok;
      } else {
        if (*(find_iter->second) < *new_tok) {
          Token::TokenDelete(find_iter->second);
          find_iter->second = new_tok;
        } else {
          Token::TokenDelete(new_tok);
        }
      }
    }
  }
  num_frames_decoded_++;
}

void SimpleDecoder::ProcessNonemitting() {
  // Processes nonemitting arcs for one frame.  Propagates within
  // cur_toks_.
  std::vector<StateId> queue;
  double infinity = std::numeric_limits<double>::infinity();
  double best_cost = infinity;
  for (const auto &kv : cur_toks_) {
    queue.push_back(kv.first);
    best_cost = std::min(best_cost, kv.second->cost_);
  }
  double cutoff = best_cost + beam_;

  while (!queue.empty()) {
    StateId state = queue.back();
    queue.pop_back();
    Token *tok = cur_toks_[state];
    if (tok == nullptr || state != tok->arc_.nextstate) {
      throw std::runtime_error("SimpleDecoder::ProcessNonemitting: bad token");
    }
    for (fst::ArcIterator<fst::Fst<StdArc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const StdArc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // propagate nonemitting only...
        continue;
      }

      const float acoustic_cost = 0.0;
      Token *new_tok = new Token(arc, acoustic_cost, tok);
      if (new_tok->cost_ > cutoff) {
        Token::TokenDelete(new_tok);
      } else {
        auto find_iter = cur_toks_.find(arc.nextstate);
        if (find_iter == cur_toks_.end()) {
          cur_toks_[arc.nextstate] = new_tok;
          queue.push_back(arc.nextstate);
        } else {
          if (*(find_iter->second) < *new_tok) {
            // find_iter has a higher cost
            Token::TokenDelete(find_iter->second);
            find_iter->second = new_tok;
            queue.push_back(arc.nextstate);
          } else {
            Token::TokenDelete(new_tok);
          }
        }
      }
    }
  }
}

// static
void SimpleDecoder::ClearToks(std::unordered_map<StateId, Token *> &toks) {
  for (auto &kv : toks) {
    Token::TokenDelete(kv.second);
  }
  toks.clear();
}

// static
void SimpleDecoder::PruneToks(float beam,
                              std::unordered_map<StateId, Token *> *toks) {
  if (toks->empty()) {
    return;
  }
  double best_cost = std::numeric_limits<double>::infinity();
  for (const auto &kv : *toks) {
    best_cost = std::min(best_cost, kv.second->cost_);
  }

  std::vector<StateId> retained;
  double cutoff = best_cost + beam;
  for (auto &kv : *toks) {
    if (kv.second->cost_ < cutoff) {
      retained.push_back(kv.first);
    } else {
      Token::TokenDelete(kv.second);
    }
  }

  std::unordered_map<StateId, Token *> tmp;
  for (StateId s : retained) {
    tmp[s] = (*toks)[s];
  }

  tmp.swap(*toks);
}

}  // namespace kaldi_decoder
