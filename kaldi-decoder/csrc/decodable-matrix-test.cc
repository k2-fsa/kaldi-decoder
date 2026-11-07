// kaldi-decoder/csrc/decodable-matrix-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Unit tests for the matrix-backed decodables with hand-computed goldens.

#include "kaldi-decoder/csrc/decodable-matrix.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-decoder/csrc/decodable-ctc.h"
#include "kaldi-decoder/csrc/transition-information.h"

namespace kaldi_decoder {

// Minimal TransitionInformation for tests: 1-based tid -> pdf map.
class FakeTransitionInformation : public TransitionInformation {
 public:
  explicit FakeTransitionInformation(std::vector<int32_t> tid_to_pdf)
      : tid_to_pdf_(std::move(tid_to_pdf)) {
    // index 0 is unused (transition-ids are one-based); pdf ids are 0-based.
    num_pdfs_ = 0;
    for (int32_t p : tid_to_pdf_) num_pdfs_ = std::max(num_pdfs_, p + 1);
  }

  bool TransitionIdsEquivalent(int32_t a, int32_t b) const override {
    return tid_to_pdf_[a] == tid_to_pdf_[b];
  }
  bool TransitionIdIsStartOfPhone(int32_t) const override { return true; }
  int32_t TransitionIdToPhone(int32_t) const override { return 1; }
  bool IsFinal(int32_t) const override { return false; }
  bool IsSelfLoop(int32_t) const override { return false; }
  const std::vector<int32_t> &TransitionIdToPdfArray() const override {
    return tid_to_pdf_;
  }
  int32_t NumPdfs() const override { return num_pdfs_; }

 private:
  std::vector<int32_t> tid_to_pdf_;
  int32_t num_pdfs_;
};

// tid 0 is a placeholder; tids 1..4 map to pdfs 0..3 (tid 2 shares pdf 1).
FakeTransitionInformation MakeFakeTm() {
  return FakeTransitionInformation({-1, 0, 1, 1, 2, 3});
}

TEST(DecodableMatrixScaledMapped, HandGolden) {
  auto tm = MakeFakeTm();
  // 2 frames x 4 pdfs.
  FloatMatrix likes(2, 4);
  likes << 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f;

  DecodableMatrixScaledMapped dec(tm, likes, 2.0f);
  EXPECT_EQ(dec.NumFramesReady(), 2);
  EXPECT_EQ(dec.NumIndices(), 5);  // tids 1..5? -> NumTransitionIds = size-1 = 5
  EXPECT_TRUE(dec.IsLastFrame(1));
  EXPECT_FALSE(dec.IsLastFrame(0));
  // LogLikelihood(frame, tid) = scale * likes(frame, pdf(tid))
  EXPECT_FLOAT_EQ(dec.LogLikelihood(0, 1), 2.0f * 0.0f);
  EXPECT_FLOAT_EQ(dec.LogLikelihood(0, 2), 2.0f * -1.0f);
  EXPECT_FLOAT_EQ(dec.LogLikelihood(1, 3), 2.0f * -5.0f);  // pdf 1 shared
  EXPECT_FLOAT_EQ(dec.LogLikelihood(1, 5), 2.0f * -7.0f);
}

TEST(DecodableMatrixMapped, FrameOffset) {
  auto tm = MakeFakeTm();
  FloatMatrix likes(1, 4);
  likes << -1.0f, -2.0f, -3.0f, -4.0f;

  // Row 0 of 'likes' is frame 3 of the utterance.
  DecodableMatrixMapped dec(tm, likes, 3);
  EXPECT_EQ(dec.NumFramesReady(), 4);
  EXPECT_TRUE(dec.IsLastFrame(3));
  EXPECT_FALSE(dec.IsLastFrame(2));
  EXPECT_FLOAT_EQ(dec.LogLikelihood(3, 1), -1.0f);
  EXPECT_FLOAT_EQ(dec.LogLikelihood(3, 2), -2.0f);
}

TEST(DecodableMatrixMappedOffset, AcceptLoglikesChunks) {
  auto tm = MakeFakeTm();
  DecodableMatrixMappedOffset dec(tm);
  EXPECT_EQ(dec.NumFramesReady(), 0);

  FloatMatrix chunk1(2, 4);
  chunk1 << 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f;
  dec.AcceptLoglikes(&chunk1, 0);
  EXPECT_EQ(dec.NumFramesReady(), 2);
  EXPECT_EQ(dec.FirstAvailableFrame(), 0);
  EXPECT_FLOAT_EQ(dec.LogLikelihood(1, 5), -7.0f);

  // Append one more frame, discarding the first stored frame.
  FloatMatrix chunk2(1, 4);
  chunk2 << -8.0f, -9.0f, -10.0f, -11.0f;
  dec.AcceptLoglikes(&chunk2, 1);
  EXPECT_EQ(dec.NumFramesReady(), 3);  // frames 1..3
  EXPECT_EQ(dec.FirstAvailableFrame(), 1);
  EXPECT_FLOAT_EQ(dec.LogLikelihood(1, 1), -4.0f);  // old frame 1 survived
  EXPECT_FLOAT_EQ(dec.LogLikelihood(2, 1), -8.0f);  // appended frame
  dec.InputIsFinished();
  EXPECT_TRUE(dec.IsLastFrame(2));
  EXPECT_FALSE(dec.IsLastFrame(1));
}

TEST(DecodableMatrixScaled, HandGolden) {
  FloatMatrix likes(2, 3);
  likes << 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f;

  DecodableMatrixScaled dec(likes, 0.5f);
  EXPECT_EQ(dec.NumFramesReady(), 2);
  EXPECT_EQ(dec.NumIndices(), 3);
  EXPECT_FLOAT_EQ(dec.LogLikelihood(0, 1), 0.5f * 0.0f);
  EXPECT_FLOAT_EQ(dec.LogLikelihood(1, 3), 0.5f * -5.0f);
}

TEST(DecodableCtc, CopyDisabled) {
  // Regression: p_ aliases the internal matrix; copying would leave a
  // dangling alias (use-after-free).  The type must not be copyable.
  static_assert(!std::is_copy_constructible<DecodableCtc>::value,
                "DecodableCtc must not be copy-constructible");
  static_assert(!std::is_copy_assignable<DecodableCtc>::value,
                "DecodableCtc must not be copy-assignable");
}

TEST(DecodableCtc, EmptyMatrixAndBounds) {
  // Regression: constructing from an EMPTY matrix used to form &mat(0,0)
  // (UB); now safe, and any query raises instead of reading out of bounds.
  FloatMatrix empty(0, 5);
  DecodableCtc dec(empty);
  EXPECT_EQ(dec.NumFramesReady(), 0);
  EXPECT_THROW(dec.LogLikelihood(0, 1), std::runtime_error);

  FloatMatrix m(2, 3);
  m << 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f;
  DecodableCtc ok(m);
  EXPECT_FLOAT_EQ(ok.LogLikelihood(1, 3), -5.0f);  // 1-based: col 2
  EXPECT_THROW(ok.LogLikelihood(2, 1), std::runtime_error);   // frame OOB
  EXPECT_THROW(ok.LogLikelihood(0, 0), std::runtime_error);   // index OOB
  EXPECT_THROW(ok.LogLikelihood(0, 4), std::runtime_error);   // index OOB
}

TEST(DecodableCtc, RawPointerCtorSharesMemory) {
  // Covers the (const float*, rows, cols, offset) ctor: shares memory, no
  // copy; queries read the caller's buffer.
  std::vector<float> buf = {0.0f, -1.0f, -2.0f, -3.0f};
  DecodableCtc dec(buf.data(), 2, 2, 0);
  EXPECT_EQ(dec.NumFramesReady(), 2);
  EXPECT_EQ(dec.NumIndices(), 2);
  EXPECT_TRUE(dec.IsLastFrame(1));
  EXPECT_FLOAT_EQ(dec.LogLikelihood(1, 2), -3.0f);
  buf[3] = -9.0f;  // shared memory: the decodable sees the update
  EXPECT_FLOAT_EQ(dec.LogLikelihood(1, 2), -9.0f);
}

TEST(DecodableMatrixMapped, FrameBoundsRaise) {
  // Regression: frame was never bounds-checked (Eigen OOB in release).
  auto tm = MakeFakeTm();
  FloatMatrix likes(1, 4);
  likes << -1.0f, -2.0f, -3.0f, -4.0f;
  DecodableMatrixScaledMapped m(tm, likes, 1.0f);
  EXPECT_THROW(m.LogLikelihood(999, 1), std::runtime_error);
  EXPECT_THROW(m.LogLikelihood(-1, 1), std::runtime_error);

  DecodableMatrixMapped mm(tm, likes, 3);  // frames 3..3
  EXPECT_FLOAT_EQ(mm.LogLikelihood(3, 1), -1.0f);
  EXPECT_THROW(mm.LogLikelihood(2, 1), std::runtime_error);  // below offset
  EXPECT_THROW(mm.LogLikelihood(4, 1), std::runtime_error);  // above end
}

}  // namespace kaldi_decoder

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
