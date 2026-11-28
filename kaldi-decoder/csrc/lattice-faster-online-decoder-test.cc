// kaldi-decoder/csrc/lattice-faster-online-decoder-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Tests for LatticeFasterOnlineDecoder.  The backpointer traceback must give
// the same best path as the shortest path through the raw lattice (the check
// kaldi's TestGetBestPath() does), at every frame of an incremental decode.

#include "kaldi-decoder/csrc/lattice-faster-online-decoder.h"

#include <random>
#include <vector>

#include "fst/fstlib.h"
#include "gtest/gtest.h"
#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/lattice-faster-decoder.h"

namespace kaldi_decoder {
namespace {

// Deterministic decodable: score table [frame][label], labels 1-based.
// num_frames_ready can be raised step by step for incremental decoding.
class TableDecodable : public DecodableInterface {
 public:
  TableDecodable(const std::vector<std::vector<float>> &scores,
                 int32_t num_indices)
      : scores_(scores),
        num_indices_(num_indices),
        num_ready_(static_cast<int32_t>(scores.size())) {}

  void SetNumFramesReady(int32_t n) { num_ready_ = n; }

  float LogLikelihood(int32_t frame, int32_t index) override {
    return scores_[frame][index - 1];
  }
  int32_t NumFramesReady() const override { return num_ready_; }
  bool IsLastFrame(int32_t frame) const override {
    return frame == static_cast<int32_t>(scores_.size()) - 1;
  }
  int32_t NumIndices() const override { return num_indices_; }

 private:
  std::vector<std::vector<float>> scores_;
  int32_t num_indices_;
  int32_t num_ready_;
};

// A small random graph with self-loops, input-epsilon arcs and several
// final states; labels in [1, num_labels].
fst::VectorFst<fst::StdArc> MakeRandomGraph(int32_t num_states,
                                            int32_t num_labels,
                                            std::mt19937 *rng) {
  std::uniform_int_distribution<int32_t> state_dist(0, num_states - 1);
  std::uniform_int_distribution<int32_t> label_dist(1, num_labels);
  std::uniform_real_distribution<float> cost_dist(0.0f, 2.0f);
  fst::VectorFst<fst::StdArc> fst;
  for (int32_t s = 0; s < num_states; s++) fst.AddState();
  fst.SetStart(0);
  for (int32_t s = 0; s < num_states; s++) {
    // self-loop
    int32_t self_label = label_dist(*rng);
    fst.AddArc(s, fst::StdArc(self_label, 0, cost_dist(*rng), s));
    for (int32_t k = 0; k < 3; k++) {
      int32_t label = label_dist(*rng);
      fst.AddArc(s, fst::StdArc(label, label, cost_dist(*rng),
                                state_dist(*rng)));
    }
    // a forward epsilon arc (no epsilon cycles: only to higher states)
    if (s + 1 < num_states) {
      fst.AddArc(s, fst::StdArc(0, 0, cost_dist(*rng), s + 1));
    }
    if (s % 3 == 2) fst.SetFinal(s, cost_dist(*rng));
  }
  return fst;
}

std::vector<int32_t> IlabelsOfLinearLattice(const fst::Lattice &lat) {
  std::vector<int32_t> ans;
  auto s = lat.Start();
  while (s != fst::kNoStateId) {
    fst::ArcIterator<fst::Lattice> aiter(lat, s);
    if (aiter.Done()) break;
    if (aiter.Value().ilabel != 0) ans.push_back(aiter.Value().ilabel);
    s = aiter.Value().nextstate;
  }
  return ans;
}

// Best path by ShortestPath over the raw lattice (kaldi's reference).
std::vector<int32_t> RawLatticeBestIlabels(
    const LatticeFasterOnlineDecoder &decoder, bool use_final_probs) {
  fst::Lattice raw;
  decoder.GetRawLattice(&raw, use_final_probs);
  fst::Lattice best;
  fst::ShortestPath(raw, &best);
  return IlabelsOfLinearLattice(best);
}

}  // namespace

TEST(LatticeFasterOnlineDecoder, TracebackMatchesRawLattice) {
  std::mt19937 rng(20261127);
  std::uniform_real_distribution<float> score_dist(-8.0f, 0.0f);
  const int32_t num_labels = 6;
  for (int32_t trial = 0; trial < 20; trial++) {
    auto graph = MakeRandomGraph(8, num_labels, &rng);
    const int32_t num_frames = 25;
    std::vector<std::vector<float>> scores(num_frames,
                                           std::vector<float>(num_labels));
    for (auto &row : scores)
      for (auto &x : row) x = score_dist(rng);
    TableDecodable decodable(scores, num_labels);

    LatticeFasterDecoderConfig config;
    config.beam = 12.0;
    config.lattice_beam = 6.0;
    LatticeFasterOnlineDecoder decoder(graph, config);
    decoder.InitDecoding();
    for (int32_t n = 1; n <= num_frames; n++) {
      decodable.SetNumFramesReady(n);
      decoder.AdvanceDecoding(&decodable);
      ASSERT_EQ(decoder.NumFramesDecoded(), n);
      std::vector<int32_t> ids = GetBestPathTransitionIds(decoder, false);
      ASSERT_EQ(static_cast<int32_t>(ids.size()), n);
      for (int32_t id : ids) EXPECT_GT(id, 0);
      EXPECT_EQ(ids, RawLatticeBestIlabels(decoder, false))
          << "trial " << trial << " frame " << n;

      fst::Lattice best;
      ASSERT_TRUE(decoder.GetBestPath(&best, false));
      EXPECT_EQ(IlabelsOfLinearLattice(best), ids);
    }
    decoder.FinalizeDecoding();
    std::vector<int32_t> ids = GetBestPathTransitionIds(decoder, true);
    EXPECT_EQ(ids, RawLatticeBestIlabels(decoder, true)) << "trial " << trial;
    EXPECT_THROW(GetBestPathTransitionIds(decoder, false), std::runtime_error);
  }
}

TEST(LatticeFasterOnlineDecoder, SameDecodeAsLatticeFasterDecoder) {
  // The backpointer tokens must not change the search.
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> score_dist(-8.0f, 0.0f);
  const int32_t num_labels = 5;
  auto graph = MakeRandomGraph(10, num_labels, &rng);
  std::vector<std::vector<float>> scores(30, std::vector<float>(num_labels));
  for (auto &row : scores)
    for (auto &x : row) x = score_dist(rng);

  LatticeFasterDecoderConfig config;
  TableDecodable d1(scores, num_labels);
  LatticeFasterDecoder plain(graph, config);
  ASSERT_TRUE(plain.Decode(&d1));
  fst::Lattice best1;
  ASSERT_TRUE(plain.GetBestPath(&best1, true));

  TableDecodable d2(scores, num_labels);
  LatticeFasterOnlineDecoder online(graph, config);
  ASSERT_TRUE(online.Decode(&d2));
  fst::Lattice best2;
  ASSERT_TRUE(online.GetBestPath(&best2, true));

  EXPECT_EQ(IlabelsOfLinearLattice(best1), IlabelsOfLinearLattice(best2));
}

TEST(LatticeFasterOnlineDecoder, GetRawLatticePruned) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> score_dist(-8.0f, 0.0f);
  const int32_t num_labels = 5;
  auto graph = MakeRandomGraph(10, num_labels, &rng);
  std::vector<std::vector<float>> scores(30, std::vector<float>(num_labels));
  for (auto &row : scores)
    for (auto &x : row) x = score_dist(rng);
  TableDecodable decodable(scores, num_labels);
  LatticeFasterDecoderConfig config;
  config.lattice_beam = 8.0;
  LatticeFasterOnlineDecoder decoder(graph, config);
  ASSERT_TRUE(decoder.Decode(&decodable));

  fst::Lattice raw;
  ASSERT_TRUE(decoder.GetRawLattice(&raw, true));
  fst::Lattice pruned_wide;
  ASSERT_TRUE(decoder.GetRawLatticePruned(&pruned_wide, true, 1e10));
  fst::Lattice pruned_tight;
  ASSERT_TRUE(decoder.GetRawLatticePruned(&pruned_tight, true, 0.5));
  // a beam wider than lattice_beam keeps every token of the raw lattice
  EXPECT_EQ(pruned_wide.NumStates(), raw.NumStates());
  EXPECT_LE(pruned_tight.NumStates(), pruned_wide.NumStates());

  // the best path survives any beam
  fst::Lattice best;
  ASSERT_TRUE(decoder.GetBestPath(&best, true));
  for (const fst::Lattice *lat : {&pruned_wide, &pruned_tight}) {
    fst::Lattice b;
    fst::ShortestPath(*lat, &b);
    EXPECT_EQ(IlabelsOfLinearLattice(b), IlabelsOfLinearLattice(best));
  }
}

TEST(LatticeFasterOnlineDecoder, NoFramesDecoded) {
  std::mt19937 rng(1);
  auto graph = MakeRandomGraph(4, 3, &rng);
  LatticeFasterDecoderConfig config;
  LatticeFasterOnlineDecoder decoder(graph, config);
  decoder.InitDecoding();
  EXPECT_TRUE(GetBestPathTransitionIds(decoder, false).empty());
}

}  // namespace kaldi_decoder
