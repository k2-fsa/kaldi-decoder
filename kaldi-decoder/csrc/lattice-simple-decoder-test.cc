// kaldi-decoder/csrc/lattice-simple-decoder-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Unit tests for LatticeSimpleDecoder with hand-computed goldens on tiny
// graphs.  (../kaldi/src/decoder/ has no decoder unit tests; binary-level
// parity against kaldi happens in kaldi-hmm-gmm's tests/parity.)

#include "kaldi-decoder/csrc/lattice-simple-decoder.h"

#include <set>
#include <vector>

#include "fst/fstlib.h"
#include "gtest/gtest.h"
#include "kaldi-decoder/csrc/decodable-itf.h"

namespace kaldi_decoder {

// Deterministic decodable: score table [frame][label], labels 1-based.
class TableDecodable : public DecodableInterface {
 public:
  TableDecodable(const std::vector<std::vector<float>> &scores,
                 int32_t num_indices)
      : scores_(scores), num_indices_(num_indices) {}

  float LogLikelihood(int32_t frame, int32_t index) override {
    if (frame < 0 || frame >= static_cast<int32_t>(scores_.size())) return -1e10;
    if (index < 1 || index > num_indices_) return -1e10;
    return scores_[frame][index - 1];
  }
  int32_t NumFramesReady() const override { return scores_.size(); }
  bool IsLastFrame(int32_t frame) const override {
    return frame == NumFramesReady() - 1;
  }
  int32_t NumIndices() const override { return num_indices_; }

 private:
  std::vector<std::vector<float>> scores_;
  int32_t num_indices_;
};

fst::VectorFst<fst::StdArc> MakeLinearGraph() {
  fst::VectorFst<fst::StdArc> fst;
  auto s0 = fst.AddState();
  auto s1 = fst.AddState();
  auto s2 = fst.AddState();
  fst.SetStart(s0);
  fst.AddArc(s0, fst::StdArc(1, 1, fst::StdArc::Weight::One(), s1));
  fst.AddArc(s1, fst::StdArc(2, 2, fst::StdArc::Weight::One(), s2));
  fst.SetFinal(s2, fst::StdArc::Weight::One());
  return fst;
}

// Two parallel 2-frame branches sharing start and final states:
//   A: s0 --1/1 [w_a]--> s1 --2/2 [w_a]--> sf (final)
//   B: s0 --3/3 [w_b]--> s2 --4/4 [w_b]--> sf
fst::VectorFst<fst::StdArc> MakeTwoBranchGraph(float w_a, float w_b) {
  fst::VectorFst<fst::StdArc> fst;
  auto s0 = fst.AddState();
  auto s1 = fst.AddState();
  auto s2 = fst.AddState();
  auto sf = fst.AddState();
  fst.SetStart(s0);
  fst.AddArc(s0, fst::StdArc(1, 1, fst::StdArc::Weight(w_a), s1));
  fst.AddArc(s1, fst::StdArc(2, 2, fst::StdArc::Weight(w_a), sf));
  fst.AddArc(s0, fst::StdArc(3, 3, fst::StdArc::Weight(w_b), s2));
  fst.AddArc(s2, fst::StdArc(4, 4, fst::StdArc::Weight(w_b), sf));
  fst.SetFinal(sf, fst::StdArc::Weight::One());
  return fst;
}

std::vector<fst::StdArc::Label> BestPathLabels(const fst::Lattice &lat) {
  std::vector<fst::StdArc::Label> labels;
  auto s = lat.Start();
  while (lat.Final(s) == fst::LatticeWeight::Zero()) {
    fst::ArcIterator<fst::Lattice> aiter(lat, s);
    EXPECT_FALSE(aiter.Done());
    labels.push_back(aiter.Value().ilabel);
    s = aiter.Value().nextstate;
  }
  return labels;
}

// Total cost of a best path: sum of arc (graph+acoustic) + final weight.
float BestPathTotalCost(const fst::Lattice &lat) {
  float total = 0.0f;
  auto s = lat.Start();
  while (true) {
    auto w = lat.Final(s);
    if (w != fst::LatticeWeight::Zero()) {
      total += w.Value1() + w.Value2();
      break;
    }
    fst::ArcIterator<fst::Lattice> aiter(lat, s);
    EXPECT_FALSE(aiter.Done());
    total += aiter.Value().weight.Value1() + aiter.Value().weight.Value2();
    s = aiter.Value().nextstate;
  }
  return total;
}

std::set<fst::StdArc::Label> AllIlabels(const fst::Lattice &lat) {
  std::set<fst::StdArc::Label> labels;
  for (fst::StateIterator<fst::Lattice> siter(lat); !siter.Done();
       siter.Next()) {
    for (fst::ArcIterator<fst::Lattice> aiter(lat, siter.Value());
         !aiter.Done(); aiter.Next()) {
      labels.insert(aiter.Value().ilabel);
    }
  }
  return labels;
}

TEST(LatticeSimpleDecoder, LinearGraph) {
  auto graph = MakeLinearGraph();
  TableDecodable dec({{0.0f, -10.0f, -10.0f, -10.0f},
                      {-10.0f, 0.0f, -10.0f, -10.0f}},
                     4);
  LatticeSimpleDecoderConfig config;
  LatticeSimpleDecoder decoder(graph, config);
  ASSERT_TRUE(decoder.Decode(&dec));
  EXPECT_EQ(decoder.NumFramesDecoded(), 2);
  EXPECT_TRUE(decoder.ReachedFinal());

  fst::Lattice best;
  ASSERT_TRUE(decoder.GetBestPath(&best, true));
  std::vector<fst::StdArc::Label> expected = {1, 2};
  EXPECT_EQ(BestPathLabels(best), expected);
}

TEST(LatticeSimpleDecoder, TwoBranchesAcoustic) {
  auto graph = MakeTwoBranchGraph(0.0f, 0.0f);
  LatticeSimpleDecoderConfig config;
  config.beam = 16.0;
  config.lattice_beam = 10.0;

  TableDecodable dec_a({{0.0f, -10.0f, -10.0f, -10.0f},
                        {-10.0f, 0.0f, -10.0f, -10.0f}},
                       4);
  {
    LatticeSimpleDecoder decoder(graph, config);
    ASSERT_TRUE(decoder.Decode(&dec_a));
    fst::Lattice best;
    ASSERT_TRUE(decoder.GetBestPath(&best, true));
    std::vector<fst::StdArc::Label> expected = {1, 2};
    EXPECT_EQ(BestPathLabels(best), expected);
  }
  TableDecodable dec_b({{-10.0f, -10.0f, 0.0f, -10.0f},
                        {-10.0f, -10.0f, -10.0f, 0.0f}},
                       4);
  {
    LatticeSimpleDecoder decoder(graph, config);
    ASSERT_TRUE(decoder.Decode(&dec_b));
    fst::Lattice best;
    ASSERT_TRUE(decoder.GetBestPath(&best, true));
    std::vector<fst::StdArc::Label> expected = {3, 4};
    EXPECT_EQ(BestPathLabels(best), expected);
  }
}

TEST(LatticeSimpleDecoder, GraphWeightGolden) {
  // Uniform acoustics + weights 0.5/2.0: branch A wins with total graph cost
  // exactly 2 * 0.5 = 1.0 (hand-computed golden).
  TableDecodable uniform({{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}},
                         4);
  auto graph = MakeTwoBranchGraph(0.5f, 2.0f);
  LatticeSimpleDecoderConfig config;
  LatticeSimpleDecoder decoder(graph, config);
  ASSERT_TRUE(decoder.Decode(&uniform));
  fst::Lattice best;
  ASSERT_TRUE(decoder.GetBestPath(&best, true));
  std::vector<fst::StdArc::Label> expected = {1, 2};
  EXPECT_EQ(BestPathLabels(best), expected);
  EXPECT_FLOAT_EQ(BestPathTotalCost(best), 1.0f);
}

TEST(LatticeSimpleDecoder, RawLatticeKeepsAlternatives) {
  // Soft scores (-1.0 gap) keep both branches inside the lattice beam, so
  // the raw lattice must contain all four labels.
  TableDecodable soft({{0.0f, -1.0f, -1.0f, -1.0f},
                       {-1.0f, 0.0f, -1.0f, -1.0f}},
                      4);
  auto graph = MakeTwoBranchGraph(0.0f, 0.0f);
  LatticeSimpleDecoderConfig config;
  LatticeSimpleDecoder decoder(graph, config);
  ASSERT_TRUE(decoder.Decode(&soft));

  fst::Lattice raw;
  ASSERT_TRUE(decoder.GetRawLattice(&raw, true));
  std::set<fst::StdArc::Label> expected = {1, 2, 3, 4};
  EXPECT_EQ(AllIlabels(raw), expected);

  // ... but the best path is still branch A.
  fst::Lattice best;
  ASSERT_TRUE(decoder.GetBestPath(&best, true));
  std::vector<fst::StdArc::Label> expected_path = {1, 2};
  EXPECT_EQ(BestPathLabels(best), expected_path);
}

}  // namespace kaldi_decoder

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
