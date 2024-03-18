// kaldi-decoder/python/csrc/kaldi-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

#include "kaldi-decoder/python/csrc/decodable-ctc.h"
#include "kaldi-decoder/python/csrc/decodable-itf.h"
#include "kaldi-decoder/python/csrc/faster-decoder.h"
#include "kaldi-decoder/python/csrc/lattice-simple-decoder.h"
#include "kaldi-decoder/python/csrc/simple-decoder.h"

namespace kaldi_decoder {

PYBIND11_MODULE(_kaldi_decoder, m) {
  m.doc() = "pybind11 binding of kaldi-decoder";
  PybindDecodableItf(&m);
  PybindFasterDecoder(&m);
  PybindLatticeSimpleDecoder(&m);
  PybindSimpleDecoder(&m);
  PybindDecodableCtc(&m);
}

}  // namespace kaldi_decoder
