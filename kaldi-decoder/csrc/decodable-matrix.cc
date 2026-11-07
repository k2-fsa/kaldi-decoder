// kaldi-decoder/csrc/decodable-matrix.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/decoder/decodable-matrix.cc

#include "kaldi-decoder/csrc/decodable-matrix.h"

namespace kaldi_decoder {

DecodableMatrixMapped::DecodableMatrixMapped(const TransitionInformation &tm,
                                             const FloatMatrix &likes,
                                             int32_t frame_offset)
    : trans_model_(tm),
      tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
      likes_(&likes),
      likes_to_delete_(nullptr),
      frame_offset_(frame_offset) {
  if (likes.cols() != tm.NumPdfs())
    KALDI_DECODER_ERR << "Mismatch, matrix has " << likes.cols()
                      << " cols but transition-model has " << tm.NumPdfs()
                      << " pdf-ids.";
}

DecodableMatrixMapped::DecodableMatrixMapped(const TransitionInformation &tm,
                                             const FloatMatrix *likes,
                                             int32_t frame_offset)
    : trans_model_(tm),
      tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
      likes_(likes),
      likes_to_delete_(likes),
      frame_offset_(frame_offset) {
  if (likes->cols() != tm.NumPdfs())
    KALDI_DECODER_ERR << "Mismatch, matrix has " << likes->cols()
                      << " cols but transition-model has " << tm.NumPdfs()
                      << " pdf-ids.";
}

float DecodableMatrixMapped::LogLikelihood(int32_t frame, int32_t tid) {
  KALDI_DECODER_PARANOID_ASSERT(tid >= 1 &&
                                tid < static_cast<int32_t>(tid_to_pdf_.size()));
  KALDI_DECODER_PARANOID_ASSERT(frame >= frame_offset_ &&
                                frame < NumFramesReady());
  int32_t pdf_id = tid_to_pdf_[tid];
  return (*likes_)(frame - frame_offset_, pdf_id);
}

int32_t DecodableMatrixMapped::NumFramesReady() const {
  return frame_offset_ + likes_->rows();
}

bool DecodableMatrixMapped::IsLastFrame(int32_t frame) const {
  KALDI_DECODER_ASSERT(frame < NumFramesReady());
  return frame == NumFramesReady() - 1;
}

// Indices are one-based!  This is for compatibility with OpenFst.
int32_t DecodableMatrixMapped::NumIndices() const {
  return trans_model_.NumTransitionIds();
}

DecodableMatrixMapped::~DecodableMatrixMapped() { delete likes_to_delete_; }

void DecodableMatrixMappedOffset::AcceptLoglikes(FloatMatrix *loglikes,
                                                 int32_t frames_to_discard) {
  if (loglikes->rows() == 0) return;
  KALDI_DECODER_ASSERT(loglikes->cols() == trans_model_.NumPdfs());
  KALDI_DECODER_ASSERT(frames_to_discard <= loglikes_.rows() &&
                       frames_to_discard >= 0);
  if (frames_to_discard == loglikes_.rows()) {
    loglikes_.swap(*loglikes);
    loglikes->resize(0, 0);
  } else {
    int32_t old_rows_kept = loglikes_.rows() - frames_to_discard;
    int32_t new_num_rows = old_rows_kept + loglikes->rows();
    FloatMatrix new_loglikes(new_num_rows, loglikes->cols());
    new_loglikes.block(0, 0, old_rows_kept, loglikes_.cols()) =
        loglikes_.block(frames_to_discard, 0, old_rows_kept, loglikes_.cols());
    new_loglikes.block(old_rows_kept, 0, loglikes->rows(), loglikes->cols()) =
        *loglikes;
    loglikes_.swap(new_loglikes);
  }
  frame_offset_ += frames_to_discard;
}

}  // namespace kaldi_decoder
