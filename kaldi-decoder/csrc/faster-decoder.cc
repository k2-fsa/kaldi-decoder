// kaldi-decoder/csrc/faster-decoder.cc

// Copyright 2009-2011 Microsoft Corporation
//           2012-2013 Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023 Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/decoder/faster-decoder.cc

#include "kaldi-decoder/csrc/faster-decoder.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "kaldi-decoder/csrc/log.h"
#include "kaldifst/csrc/remove-eps-local.h"

namespace kaldi_decoder {

FasterDecoder::FasterDecoder(const fst::Fst<fst::StdArc> &fst,
                             const FasterDecoderOptions &opts)
    : fst_(fst), config_(opts), num_frames_decoded_(-1) {
  KALDI_DECODER_ASSERT(config_.hash_ratio >=
                       1.0);  // less doesn't make much sense.
  KALDI_DECODER_ASSERT(config_.max_active > 1);
  KALDI_DECODER_ASSERT(config_.min_active >= 0 &&
                       config_.min_active < config_.max_active);

  // just so on the first frame we do something reasonable.
  toks_.SetSize(1000);
}

void FasterDecoder::ClearToks(Elem *list) {
  for (Elem *e = list, *e_tail; e != nullptr; e = e_tail) {
    Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

void FasterDecoder::InitDecoding() {
  // clean up from last time:
  ClearToks(toks_.Clear());
  StateId start_state = fst_.Start();

  KALDI_DECODER_ASSERT(start_state != fst::kNoStateId);

  Arc dummy_arc(0, 0, Weight::One(), start_state);

  toks_.Insert(start_state, new Token(dummy_arc, nullptr));

  ProcessNonemitting(std::numeric_limits<float>::max());

  num_frames_decoded_ = 0;
}

// TODO(dan): first time we go through this, could avoid using the queue.
void FasterDecoder::ProcessNonemitting(double cutoff) {
  // Processes nonemitting arcs for one frame.
  KALDI_DECODER_ASSERT(queue_.empty());

  for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
    queue_.push_back(e);
  }

  while (!queue_.empty()) {
    const Elem *e = queue_.back();
    queue_.pop_back();

    StateId state = e->key;
    Token *tok = e->val;  // would segfault if state not
    // in toks_ but this can't happen.
    if (tok->cost_ > cutoff) {  // Don't bother processing successors.
      continue;
    }

    KALDI_DECODER_ASSERT(tok != nullptr && state == tok->arc_.nextstate);

    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();

      if (arc.ilabel != 0) {
        continue;
      }

      // propagate nonemitting only...

      Token *new_tok = new Token(arc, tok);

      if (new_tok->cost_ > cutoff) {  // prune
        Token::TokenDelete(new_tok);
        continue;
      }

      Elem *e_found = toks_.Insert(arc.nextstate, new_tok);
      if (e_found->val == new_tok) {
        // if inserted successfully
        queue_.push_back(e_found);
        continue;
      }

      // there is another token at this state, we need
      // to compare their costs and keep the one with a lower cost

      if (*(e_found->val) < *new_tok) {
        // i.e., if the cost of e_found is larger than new_tok
        // we keep the token with a lower cost
        Token::TokenDelete(e_found->val);
        e_found->val = new_tok;
        queue_.push_back(e_found);
      } else {
        // the new token has a higher cost, remove it
        Token::TokenDelete(new_tok);
      }
    }
  }
}

void FasterDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();
  AdvanceDecoding(decodable);
}

void FasterDecoder::AdvanceDecoding(DecodableInterface *decodable,
                                    int32_t max_num_frames /*=-1*/) {
  KALDI_DECODER_ASSERT(num_frames_decoded_ >= 0 &&
                       "You must call InitDecoding() before AdvanceDecoding()");

  int32_t num_frames_ready = decodable->NumFramesReady();

  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_DECODER_ASSERT(num_frames_ready >= num_frames_decoded_);

  int32_t target_frames_decoded = num_frames_ready;

  if (max_num_frames >= 0) {
    target_frames_decoded =
        std::min(target_frames_decoded, num_frames_decoded_ + max_num_frames);
  }

  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    double weight_cutoff = ProcessEmitting(decodable);

    ProcessNonemitting(weight_cutoff);
  }
}

// ProcessEmitting returns the likelihood cutoff used.
double FasterDecoder::ProcessEmitting(DecodableInterface *decodable) {
  int32_t frame = num_frames_decoded_;
  Elem *last_toks = toks_.Clear();
  size_t tok_cnt;
  float adaptive_beam;
  Elem *best_elem = nullptr;
  double weight_cutoff =
      GetCutoff(last_toks, &tok_cnt, &adaptive_beam, &best_elem);

  // KALDI_DECODER_LOG << tok_cnt << " tokens active.";

  // This makes sure the hash is always big enough.
  PossiblyResizeHash(tok_cnt);

  // This is the cutoff we use after adding in the log-likes (i.e.
  // for the next frame).  This is a bound on the cutoff we will use
  // on the next frame.
  double next_weight_cutoff = std::numeric_limits<double>::infinity();

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // we'd propagate..
        float ac_cost = -1 * decodable->LogLikelihood(frame, arc.ilabel);
        double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;
        if (new_weight + adaptive_beam < next_weight_cutoff)
          next_weight_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // int32_t n = 0, np = 0;

  // the tokens are now owned here, in last_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call TokenDelete
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = last_toks, *e_tail; e != nullptr;
       e = e_tail) {  // loop this way
    // n++;
    // because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    if (tok->cost_ < weight_cutoff) {  // not pruned.
      // np++;
      KALDI_DECODER_ASSERT(state == tok->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, state); !aiter.Done();
           aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          float ac_cost = -1 * decodable->LogLikelihood(frame, arc.ilabel);
          double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;
          if (new_weight < next_weight_cutoff) {  // not pruned..
            Token *new_tok = new Token(arc, ac_cost, tok);
            Elem *e_found = toks_.Insert(arc.nextstate, new_tok);

            if (new_weight + adaptive_beam < next_weight_cutoff) {
              next_weight_cutoff = new_weight + adaptive_beam;
            }

            if (e_found->val != new_tok) {
              if (*(e_found->val) < *new_tok) {
                // e_found has a higher cost
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {
                // new_tok has a higher cost
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }

    e_tail = e->tail;
    Token::TokenDelete(e->val);
    toks_.Delete(e);
  }

  num_frames_decoded_++;
  return next_weight_cutoff;
}

// Gets the weight cutoff.  Also counts the active tokens.
double FasterDecoder::GetCutoff(Elem *list_head, size_t *tok_count,
                                float *adaptive_beam, Elem **best_elem) {
  double best_cost = std::numeric_limits<double>::infinity();

  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32_t>::max() &&
      config_.min_active == 0) {
    // no constraints
    for (Elem *e = list_head; e != nullptr; e = e->tail, ++count) {
      double w = e->val->cost_;
      if (w < best_cost) {
        best_cost = w;

        if (best_elem) {
          *best_elem = e;
        }
      }
    }

    if (tok_count != nullptr) {
      *tok_count = count;
    }

    if (adaptive_beam != nullptr) {
      *adaptive_beam = config_.beam;
    }

    return best_cost + config_.beam;
  }

  tmp_array_.clear();

  for (Elem *e = list_head; e != nullptr; e = e->tail, ++count) {
    double w = e->val->cost_;
    tmp_array_.push_back(w);

    if (w < best_cost) {
      best_cost = w;

      if (best_elem) {
        *best_elem = e;
      }
    }
  }

  if (tok_count != nullptr) {
    *tok_count = count;
  }

  double beam_cutoff = best_cost + config_.beam;
  double min_active_cutoff = std::numeric_limits<double>::infinity();
  double max_active_cutoff = std::numeric_limits<double>::infinity();

  if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
    std::nth_element(tmp_array_.begin(),
                     tmp_array_.begin() + config_.max_active, tmp_array_.end());
    max_active_cutoff = tmp_array_[config_.max_active];
  }

  if (max_active_cutoff < beam_cutoff) {  // max_active is tighter than beam.
    if (adaptive_beam) {
      *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta;
    }

    return max_active_cutoff;
  }

  if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
    if (config_.min_active == 0) {
      min_active_cutoff = best_cost;
    } else {
      std::nth_element(
          tmp_array_.begin(), tmp_array_.begin() + config_.min_active,
          tmp_array_.size() > static_cast<size_t>(config_.max_active)
              ? tmp_array_.begin() + config_.max_active
              : tmp_array_.end());

      min_active_cutoff = tmp_array_[config_.min_active];
    }
  }

  if (min_active_cutoff > beam_cutoff) {  // min_active is looser than beam.
    if (adaptive_beam) {
      *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta;
    }

    return min_active_cutoff;
  } else {
    *adaptive_beam = config_.beam;

    return beam_cutoff;
  }
}

void FasterDecoder::PossiblyResizeHash(size_t num_toks) {
  auto new_sz =
      static_cast<size_t>(static_cast<float>(num_toks) * config_.hash_ratio);

  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

bool FasterDecoder::ReachedFinal() const {
  for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
    if (e->val->cost_ != std::numeric_limits<double>::infinity() &&
        fst_.Final(e->key) != Weight::Zero())
      return true;
  }
  return false;
}

bool FasterDecoder::GetBestPath(fst::MutableFst<fst::LatticeArc> *fst_out,
                                bool use_final_probs) {
  // GetBestPath gets the decoding output.  If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  fst_out->DeleteStates();
  Token *best_tok = nullptr;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
      if (best_tok == nullptr || *best_tok < *(e->val)) {
        best_tok = e->val;
      }
    }
  } else {
    double infinity = std::numeric_limits<double>::infinity();
    double best_cost = infinity;

    for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
      double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }

  if (best_tok == nullptr) {
    // No output.
    return false;
  }

  std::vector<fst::LatticeArc> arcs_reverse;  // arcs in reverse order.

  for (Token *tok = best_tok; tok != nullptr; tok = tok->prev_) {
    float tot_cost = tok->cost_ - (tok->prev_ ? tok->prev_->cost_ : 0.0);
    float graph_cost = tok->arc_.weight.Value();
    float ac_cost = tot_cost - graph_cost;

    fst::LatticeArc l_arc(tok->arc_.ilabel, tok->arc_.olabel,
                          fst::LatticeWeight(graph_cost, ac_cost),
                          tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }

  KALDI_DECODER_ASSERT(arcs_reverse.back().nextstate == fst_.Start());

  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size()) - 1; i >= 0; --i) {
    fst::LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs) {
    Weight final_weight = fst_.Final(best_tok->arc_.nextstate);
    fst_out->SetFinal(cur_state, fst::LatticeWeight(final_weight.Value(), 0.0));
  } else {
    fst_out->SetFinal(cur_state, fst::LatticeWeight::One());
  }
  fst::RemoveEpsLocal(fst_out);
  return true;
}

}  // namespace kaldi_decoder
