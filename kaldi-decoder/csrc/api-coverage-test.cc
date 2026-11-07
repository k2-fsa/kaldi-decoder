// kaldi-decoder/csrc/api-coverage-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Coverage-driven tests: every public function that the per-decoder tests
// leave uncovered (per llvm-cov function coverage).  Each test documents
// which function it covers.

#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-decoder/csrc/eigen.h"
#include "kaldi-decoder/csrc/decoder-wrappers.h"
#include "kaldi-decoder/csrc/determinize-lattice-pruned.h"
#include "kaldi-decoder/csrc/faster-decoder.h"
#include "kaldi-decoder/csrc/lattice-faster-decoder.h"
#include "kaldi-decoder/csrc/lattice-simple-decoder.h"
#include "kaldi-decoder/csrc/simple-decoder.h"
#include "kaldi-decoder/csrc/transition-information.h"

#include <limits>
#include <stdexcept>
#include <vector>

#include "fst/fstlib.h"
#include "gtest/gtest.h"

namespace kaldi_decoder {

namespace {

class TableDecodable : public DecodableInterface {
 public:
  TableDecodable(const std::vector<std::vector<float>> &scores,
                 int32_t num_indices)
      : scores_(scores), num_indices_(num_indices) {}
  float LogLikelihood(int32_t frame, int32_t index) override {
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

class MinimalTm : public TransitionInformation {
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

// s0 --1/1--> s1 --2/2--> s2(final); word olabels 10/20.
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

TableDecodable GoodScores() {
  return TableDecodable({{0.0f, -10.0f}, {-10.0f, 0.0f}}, 2);
}

}  // namespace

// ---------------------------------------------------------------------------
// FinalRelativeCost (all three decoders) + NumFramesDecoded accessors
// ---------------------------------------------------------------------------
TEST(ApiCoverage, SimpleDecoderFinalRelativeCostAndNumFrames) {
  auto graph = MakeLinearWordGraph();
  auto dec = GoodScores();
  SimpleDecoder decoder(graph, 16.0);
  decoder.Decode(&dec);
  EXPECT_EQ(decoder.NumFramesDecoded(), 2);
  // Reached final: the relative cost is finite (== 0 here: the best token
  // is on a final state).
  EXPECT_LT(decoder.FinalRelativeCost(),
            std::numeric_limits<float>::infinity());

  // No final state: FinalRelativeCost is infinity (kaldi semantics).
  for (fst::StateIterator<fst::VectorFst<fst::StdArc>> siter(graph);
       !siter.Done(); siter.Next()) {
    graph.SetFinal(siter.Value(), fst::StdArc::Weight::Zero());
  }
  SimpleDecoder decoder2(graph, 16.0);
  decoder2.Decode(&dec);
  EXPECT_EQ(decoder2.FinalRelativeCost(),
            std::numeric_limits<float>::infinity());
}

TEST(ApiCoverage, FasterDecoderSetOptions) {
  auto graph = MakeLinearWordGraph();
  auto dec = GoodScores();
  FasterDecoderOptions opts;
  opts.beam = 16.0;
  FasterDecoder decoder(graph, opts);
  decoder.InitDecoding();
  decoder.AdvanceDecoding(&dec);
  EXPECT_EQ(decoder.NumFramesDecoded(), 2);
  EXPECT_TRUE(decoder.ReachedFinal());

  // FasterDecoder exposes SetOptions only (kaldi's API); a re-decode after
  // the change must still work.
  FasterDecoderOptions opts2;
  opts2.beam = 7.5;
  decoder.SetOptions(opts2);
  decoder.InitDecoding();
  decoder.AdvanceDecoding(&dec);
  fst::Lattice best;
  EXPECT_TRUE(decoder.GetBestPath(&best, true));
}

TEST(ApiCoverage, LatticeDecodersFinalRelativeCostAndOptions) {
  auto graph = MakeLinearWordGraph();
  auto dec = GoodScores();
  {
    LatticeSimpleDecoderConfig cfg;
    LatticeSimpleDecoder decoder(graph, cfg);
    decoder.Decode(&dec);
    EXPECT_LT(decoder.FinalRelativeCost(),
              std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(decoder.GetOptions().beam, 16.0f);
  }
  {
    LatticeFasterDecoderConfig cfg;
    LatticeFasterDecoder decoder(graph, cfg);
    decoder.Decode(&dec);
    EXPECT_LT(decoder.FinalRelativeCost(),
              std::numeric_limits<float>::infinity());
    LatticeFasterDecoderConfig cfg2;
    cfg2.lattice_beam = 3.0;
    decoder.SetOptions(cfg2);
    EXPECT_FLOAT_EQ(decoder.GetOptions().lattice_beam, 3.0f);
  }
}

// ---------------------------------------------------------------------------
// BackpointerToken instantiation + alternate FST types
// ---------------------------------------------------------------------------
TEST(ApiCoverage, BackpointerTokenDecoder) {
  auto graph = MakeLinearWordGraph();
  auto dec = GoodScores();
  LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>,
                          decoder::BackpointerToken>
      decoder(graph, LatticeFasterDecoderConfig());
  decoder.InitDecoding();
  decoder.AdvanceDecoding(&dec);
  EXPECT_TRUE(decoder.ReachedFinal());
  fst::Lattice best;
  ASSERT_TRUE(decoder.GetBestPath(&best, true));
  EXPECT_EQ(best.NumStates(), 3u);
}

TEST(ApiCoverage, ConstFstAndVectorFstInstantiations) {
  auto vgraph = MakeLinearWordGraph();
  fst::ConstFst<fst::StdArc> cgraph(vgraph);
  auto dec = GoodScores();
  {
    LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, decoder::StdToken>
        decoder(vgraph, LatticeFasterDecoderConfig());
    decoder.Decode(&dec);
    fst::Lattice best;
    EXPECT_TRUE(decoder.GetBestPath(&best, true));
  }
  {
    LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, decoder::StdToken>
        decoder(cgraph, LatticeFasterDecoderConfig());
    decoder.Decode(&dec);
    fst::Lattice best;
    EXPECT_TRUE(decoder.GetBestPath(&best, true));
    fst::Lattice raw;
    EXPECT_TRUE(decoder.GetRawLattice(&raw, true));
  }
}

// ---------------------------------------------------------------------------
// ModifyGraphForCarefulAlignment + careful/retry alignment paths
// ---------------------------------------------------------------------------
TEST(ApiCoverage, CarefulAndRetryAlignment) {
  auto dec = GoodScores();
  {
    // careful=true must still align the easy utterance.
    auto graph = MakeLinearWordGraph();
    AlignConfig cfg;
    cfg.beam = 200.0;
    cfg.retry_beam = 0.0;
    cfg.careful = true;
    int32_t num_done = 0;
    int32_t num_error = 0;
    int32_t num_retried = 0;
    double tot_like = 0;
    int64_t frame_count = 0;
    std::vector<int32_t> alignment;
    std::vector<int32_t> words;
    AlignUtteranceWrapper(cfg, "utt", 1.0f, &graph, &dec, &num_done,
                          &num_error, &num_retried, &tot_like, &frame_count,
                          &alignment, &words);
    EXPECT_EQ(num_done, 1);
    EXPECT_EQ(num_error, 0);
    std::vector<int32_t> expected_ali = {1, 2};
    EXPECT_EQ(alignment, expected_ali);
  }
  {
    // A graph with no final state + tiny beam: alignment fails on the first
    // attempt; retry_beam triggers the retry path (still fails -> error).
    auto graph = MakeLinearWordGraph();
    for (fst::StateIterator<fst::VectorFst<fst::StdArc>> siter(graph);
         !siter.Done(); siter.Next()) {
      graph.SetFinal(siter.Value(), fst::StdArc::Weight::Zero());
    }
    AlignConfig cfg;
    cfg.beam = 8.0;
    cfg.retry_beam = 16.0;
    cfg.careful = false;
    int32_t num_done = 0;
    int32_t num_error = 0;
    int32_t num_retried = 0;
    double tot_like = 0;
    int64_t frame_count = 0;
    std::vector<int32_t> alignment;
    std::vector<int32_t> words;
    AlignUtteranceWrapper(cfg, "utt", 1.0f, &graph, &dec, &num_done,
                          &num_error, &num_retried, &tot_like, &frame_count,
                          &alignment, &words);
    EXPECT_EQ(num_done, 0);
    EXPECT_EQ(num_error, 1);
  }
}

// ---------------------------------------------------------------------------
// DecodeUtteranceLatticeFaster template instantiations (VectorFst/ConstFst)
// ---------------------------------------------------------------------------
TEST(ApiCoverage, DecodeWrapperTemplateInstantiations) {
  auto vgraph = MakeLinearWordGraph();
  fst::ConstFst<fst::StdArc> cgraph(vgraph);
  auto dec = GoodScores();
  MinimalTm tm;
  {
    LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, decoder::StdToken>
        decoder(vgraph, LatticeFasterDecoderConfig());
    std::vector<int32_t> ali, words;
    double like = 0;
    fst::Lattice lat;
    EXPECT_TRUE(DecodeUtteranceLatticeFaster(decoder, dec, tm, "u", false,
                                             1.0f, &ali, &words, &like,
                                             &lat));
    std::vector<int32_t> expected_ali = {1, 2};
    EXPECT_EQ(ali, expected_ali);
  }
  {
    LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, decoder::StdToken>
        decoder(cgraph, LatticeFasterDecoderConfig());
    std::vector<int32_t> ali, words;
    double like = 0;
    EXPECT_TRUE(DecodeUtteranceLatticeFaster(decoder, dec, tm, "u", false,
                                             1.0f, &ali, &words, &like,
                                             nullptr));
    std::vector<int32_t> expected_ali = {1, 2};
    EXPECT_EQ(ali, expected_ali);
  }
}

// ---------------------------------------------------------------------------
// Randn helpers (eigen.cc utilities)
// ---------------------------------------------------------------------------
TEST(ApiCoverage, RandnHelpers) {
  FloatMatrix m = RandnMatrix(3, 4);
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 4);
  FloatVector v = RandnVector(5);
  EXPECT_EQ(v.size(), 5);
  float x = Randn();
  (void)x;
}

// ---------------------------------------------------------------------------
// Config ToString()/Check() + options structs
// ---------------------------------------------------------------------------
TEST(ApiCoverage, ConfigToStringAndCheck) {
  EXPECT_NE(LatticeFasterDecoderConfig().ToString().find("beam="),
            std::string::npos);
  EXPECT_NE(LatticeSimpleDecoderConfig().ToString().find("beam="),
            std::string::npos);
  EXPECT_NE(FasterDecoderOptions().ToString().find("beam="),
            std::string::npos);
  EXPECT_NE(DeterminizeLatticePrunedOptions().ToString().find("delta="),
            std::string::npos);
  EXPECT_NE(DeterminizeLatticePhonePrunedOptions().ToString().find("delta="),
            std::string::npos);

  LatticeFasterDecoderConfig bad;
  bad.beam = -1.0;
  EXPECT_THROW(bad.Check(), std::runtime_error);
  LatticeFasterDecoderConfig good;
  EXPECT_NO_THROW(good.Check());
}

// ---------------------------------------------------------------------------
// DecodableInterface default NumFramesReady (error path)
// ---------------------------------------------------------------------------
TEST(ApiCoverage, DecodableInterfaceDefaultNumFramesReadyThrows) {
  struct NoNumFrames : public DecodableInterface {
    float LogLikelihood(int32_t, int32_t) override { return 0.0; }
    bool IsLastFrame(int32_t) const override { return true; }
    int32_t NumIndices() const override { return 1; }
  };
  NoNumFrames d;
  EXPECT_THROW(d.NumFramesReady(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// TransitionInformation::TransitionIdToPdf (inline derived)
// ---------------------------------------------------------------------------
TEST(ApiCoverage, TransitionIdToPdf) {
  MinimalTm tm;
  EXPECT_EQ(tm.TransitionIdToPdf(1), 0);
  EXPECT_EQ(tm.TransitionIdToPdf(3), 2);
  EXPECT_EQ(tm.NumTransitionIds(), 4);
}

}  // namespace kaldi_decoder

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
