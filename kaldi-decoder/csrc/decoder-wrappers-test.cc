// kaldi-decoder/csrc/decoder-wrappers-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Unit tests for the utterance-level alignment/decoding wrappers with
// hand-computed goldens.  (Binary-level parity against kaldi's
// gmm-align-compiled / gmm-latgen-faster happens in kaldi-hmm-gmm's
// tests/parity.)

#include "kaldi-decoder/csrc/decoder-wrappers.h"

#include <string>
#include <vector>

#include "fst/fstlib.h"
#include "gtest/gtest.h"
#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/transition-information.h"

namespace kaldi_decoder {

namespace {

class TableDecodable : public DecodableInterface {
 public:
  TableDecodable(const std::vector<std::vector<float>> &scores,
                 int32_t num_indices)
      : scores_(scores), num_indices_(num_indices) {}

  float LogLikelihood(int32_t frame, int32_t index) override {
    if (frame < 0 || frame >= static_cast<int32_t>(scores_.size()))
      return -1e10;
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

class FakeTransitionInformation : public TransitionInformation {
 public:
  bool TransitionIdsEquivalent(int32_t a, int32_t b) const override {
    return a == b;
  }
  bool TransitionIdIsStartOfPhone(int32_t) const override { return true; }
  int32_t TransitionIdToPhone(int32_t) const override { return 1; }
  bool IsFinal(int32_t) const override { return false; }
  bool IsSelfLoop(int32_t) const override { return false; }
  const std::vector<int32_t> &TransitionIdToPdfArray() const override {
    return tid_to_pdf_;
  }
  int32_t NumPdfs() const override { return 4; }

 private:
  std::vector<int32_t> tid_to_pdf_ = {-1, 0, 1, 2, 3};
};

// Linear graph with distinct word olabels: s0 --1/10--> s1 --2/20--> s2.
fst::VectorFst<fst::StdArc> MakeLinearWordGraph() {
  fst::VectorFst<fst::StdArc> fst;
  auto s0 = fst.AddState();
  auto s1 = fst.AddState();
  auto s2 = fst.AddState();
  fst.SetStart(s0);
  fst.AddArc(s0, fst::StdArc(1, 10, fst::StdArc::Weight::One(), s1));
  fst.AddArc(s1, fst::StdArc(2, 20, fst::StdArc::Weight::One(), s2));
  fst.SetFinal(s2, fst::StdArc::Weight::One());
  return fst;
}

}  // namespace

TEST(DecoderWrappers, AlignUtteranceWrapperLinear) {
  auto graph = MakeLinearWordGraph();
  TableDecodable dec({{0.0f, -10.0f}, {-10.0f, 0.0f}}, 2);
  AlignConfig config;
  int32_t num_done = 0;
  int32_t num_error = 0;
  int32_t num_retried = 0;
  double tot_like = 0;
  int64_t frame_count = 0;
  std::vector<int32_t> alignment;
  std::vector<int32_t> words;
  AlignUtteranceWrapper(config, "utt", 1.0f, &graph, &dec, &num_done,
                        &num_error, &num_retried, &tot_like, &frame_count,
                        &alignment, &words);
  EXPECT_EQ(num_done, 1);
  EXPECT_EQ(num_error, 0);
  std::vector<int32_t> expected_ali = {1, 2};
  std::vector<int32_t> expected_words = {10, 20};
  EXPECT_EQ(alignment, expected_ali);
  EXPECT_EQ(words, expected_words);
  EXPECT_EQ(frame_count, 2);
}

TEST(DecoderWrappers, DecodeUtteranceLatticeFasterLinear) {
  auto graph = MakeLinearWordGraph();
  TableDecodable dec({{0.0f, -10.0f}, {-10.0f, 0.0f}}, 2);
  FakeTransitionInformation tm;
  LatticeFasterDecoderConfig config;
  LatticeFasterDecoder decoder(graph, config);

  std::vector<int32_t> alignments;
  std::vector<int32_t> words;
  double log_like = 0;
  bool succeeded = DecodeUtteranceLatticeFaster(decoder, dec, tm, "utt",
                                                /*allow_partial=*/false,
                                                /*acoustic_scale=*/1.0f,
                                                &alignments, &words, &log_like);
  EXPECT_TRUE(succeeded);
  std::vector<int32_t> expected_ali = {1, 2};
  std::vector<int32_t> expected_words = {10, 20};
  EXPECT_EQ(alignments, expected_ali);
  EXPECT_EQ(words, expected_words);
  EXPECT_NEAR(log_like, 0.0, 1.0e-5);
}

TEST(DecoderWrappers, DecodeUtteranceLatticeSimpleLinear) {
  auto graph = MakeLinearWordGraph();
  TableDecodable dec({{0.0f, -10.0f}, {-10.0f, 0.0f}}, 2);
  FakeTransitionInformation tm;
  LatticeSimpleDecoderConfig config;
  LatticeSimpleDecoder decoder(graph, config);

  std::vector<int32_t> alignments;
  std::vector<int32_t> words;
  double log_like = 0;
  bool succeeded = DecodeUtteranceLatticeSimple(decoder, dec, tm, "utt",
                                                /*allow_partial=*/false,
                                                /*acoustic_scale=*/1.0f,
                                                &alignments, &words, &log_like);
  EXPECT_TRUE(succeeded);
  std::vector<int32_t> expected_ali = {1, 2};
  std::vector<int32_t> expected_words = {10, 20};
  EXPECT_EQ(alignments, expected_ali);
  EXPECT_EQ(words, expected_words);
}

TEST(DecoderWrappers, LatticeBestPathScales) {
  // Two-path lattice: path A costs (G=2, A=0), path B costs (G=0, A=2).
  // lattice-best-path minimizes lm_scale*G + acoustic_scale*A.
  fst::Lattice lat;
  auto s0 = lat.AddState();
  auto sa = lat.AddState();
  auto sb = lat.AddState();
  auto sf = lat.AddState();
  lat.SetStart(s0);
  lat.AddArc(s0, fst::LatticeArc(1, 10, fst::LatticeWeight(2.0f, 0.0f), sa));
  lat.AddArc(sa, fst::LatticeArc(2, 20, fst::LatticeWeight(0.0f, 0.0f), sf));
  lat.AddArc(s0, fst::LatticeArc(3, 30, fst::LatticeWeight(0.0f, 0.0f), sb));
  lat.AddArc(sb, fst::LatticeArc(4, 40, fst::LatticeWeight(0.0f, 2.0f), sf));
  lat.SetFinal(sf, fst::LatticeWeight::One());

  std::vector<int32_t> alignments;
  std::vector<int32_t> words;
  double like = 0;

  // lm_scale=1, acoustic_scale=1: A = 2.0, B = 2.0 -> tie; LatticeWeight
  // Compare breaks ties by smaller Value1, so A (G=2) vs B (G=0) -> B wins.
  LatticeBestPath(lat, 1.0f, 1.0f, &alignments, &words, &like);
  std::vector<int32_t> expected_ali_b = {3, 4};
  std::vector<int32_t> expected_words_b = {30, 40};
  EXPECT_EQ(alignments, expected_ali_b);
  EXPECT_EQ(words, expected_words_b);

  // lm_scale=2: A = 4.0, B = 2.0 -> B still; acoustic_scale=2: A=2, B=4 -> A.
  LatticeBestPath(lat, 1.0f, 2.0f, &alignments, &words, &like);
  std::vector<int32_t> expected_ali_a = {1, 2};
  std::vector<int32_t> expected_words_a = {10, 20};
  EXPECT_EQ(alignments, expected_ali_a);
  EXPECT_EQ(words, expected_words_a);
  EXPECT_NEAR(like, -2.0, 1.0e-5);  // -(1*2 + 2*0)
}

}  // namespace kaldi_decoder

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
