// kaldi-decoder/python/csrc/kaldi-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

namespace kaldi_decoder {

PYBIND11_MODULE(_kaldi_decoder, m) {
  m.doc() = "pybind11 binding of kaldi-decoder";
}

} // namespace kaldi_decoder
