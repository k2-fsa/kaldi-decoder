// kaldi-decoder/csrc/simple-decoder-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Unit tests for SimpleDecoder with hand-computed goldens on tiny graphs.
// (../kaldi/src/decoder/ has no decoder unit tests; binary-level parity
// against kaldi happens in kaldi-hmm-gmm's tests/parity.)

#include "kaldi-decoder/csrc/simple-decoder.h"

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

// Linear graph: s0 --1/1--> s1 --2/2--> s2 (final).  Optionally set a final
// weight so the graph has no final state at all.
fst::VectorFst<fst::StdArc> MakeLinearGraph(bool with_final = true) {
  fst::VectorFst<fst::StdArc> fst;
  auto s0 = fst.AddState();
  auto s1 = fst.AddState();
  auto s2 = fst.AddState();
  fst.SetStart(s0);
  fst.AddArc(s0, fst::StdArc(1, 1, fst::StdArc::Weight::One(), s1));
  fst.AddArc(s1, fst::StdArc(2, 2, fst::StdArc::Weight::One(), s2));
  if (with_final) fst.SetFinal(s2, fst::StdArc::Weight::One());
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

TEST(SimpleDecoder, LinearGraph) {
  auto graph = MakeLinearGraph();
  // Correct label at frame t is t+1.
  TableDecodable dec({{0.0f, -10.0f, -10.0f, -10.0f},
                      {-10.0f, 0.0f, -10.0f, -10.0f}},
                     4);
  SimpleDecoder decoder(graph, 16.0);
  EXPECT_TRUE(decoder.Decode(&dec));
  EXPECT_TRUE(decoder.ReachedFinal());

  fst::Lattice best;
  ASSERT_TRUE(decoder.GetBestPath(&best, true));
  std::vector<fst::StdArc::Label> expected = {1, 2};
  EXPECT_EQ(BestPathLabels(best), expected);
}

TEST(SimpleDecoder, TwoBranchesAcoustic) {
  auto graph = MakeTwoBranchGraph(0.0f, 0.0f);
  // Favor branch A.
  TableDecodable dec_a({{0.0f, -10.0f, -10.0f, -10.0f},
                        {-10.0f, 0.0f, -10.0f, -10.0f}},
                       4);
  {
    SimpleDecoder decoder(graph, 16.0);
    ASSERT_TRUE(decoder.Decode(&dec_a));
    fst::Lattice best;
    ASSERT_TRUE(decoder.GetBestPath(&best, true));
    std::vector<fst::StdArc::Label> expected = {1, 2};
    EXPECT_EQ(BestPathLabels(best), expected);
  }
  // Favor branch B: the decoded path must flip.
  TableDecodable dec_b({{-10.0f, -10.0f, 0.0f, -10.0f},
                        {-10.0f, -10.0f, -10.0f, 0.0f}},
                       4);
  {
    SimpleDecoder decoder(graph, 16.0);
    ASSERT_TRUE(decoder.Decode(&dec_b));
    fst::Lattice best;
    ASSERT_TRUE(decoder.GetBestPath(&best, true));
    std::vector<fst::StdArc::Label> expected = {3, 4};
    EXPECT_EQ(BestPathLabels(best), expected);
  }
}

TEST(SimpleDecoder, TwoBranchesGraphWeight) {
  // Uniform acoustics: the cheaper branch must win.
  TableDecodable uniform({{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}},
                         4);
  {
    auto graph = MakeTwoBranchGraph(0.5f, 2.0f);
    SimpleDecoder decoder(graph, 16.0);
    ASSERT_TRUE(decoder.Decode(&uniform));
    fst::Lattice best;
    ASSERT_TRUE(decoder.GetBestPath(&best, true));
    std::vector<fst::StdArc::Label> expected = {1, 2};
    EXPECT_EQ(BestPathLabels(best), expected);
  }
  {
    auto graph = MakeTwoBranchGraph(2.0f, 0.5f);
    SimpleDecoder decoder(graph, 16.0);
    ASSERT_TRUE(decoder.Decode(&uniform));
    fst::Lattice best;
    ASSERT_TRUE(decoder.GetBestPath(&best, true));
    std::vector<fst::StdArc::Label> expected = {3, 4};
    EXPECT_EQ(BestPathLabels(best), expected);
  }
}

TEST(SimpleDecoder, NoFinalState) {
  auto graph = MakeLinearGraph(/*with_final=*/false);
  TableDecodable dec({{0.0f, -10.0f}, {-10.0f, 0.0f}}, 2);
  SimpleDecoder decoder(graph, 16.0);
  decoder.Decode(&dec);
  EXPECT_FALSE(decoder.ReachedFinal());
  fst::Lattice best;
  // kaldi semantics: with no final state GetBestPath falls back to the best
  // PARTIAL path and still returns true (see SimpleDecoder::GetBestPath).
  EXPECT_TRUE(decoder.GetBestPath(&best, true));
  std::vector<fst::StdArc::Label> expected = {1, 2};
  EXPECT_EQ(BestPathLabels(best), expected);
}

}  // namespace kaldi_decoder

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
