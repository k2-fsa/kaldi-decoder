// kaldi-decoder/python/csrc/kaldi-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

#include "kaldi-decoder/python/csrc/decodable-ctc.h"
#include "kaldi-decoder/python/csrc/decodable-itf.h"
#include "kaldi-decoder/python/csrc/decodable-matrix.h"
#include "kaldi-decoder/python/csrc/decoder-wrappers.h"
#include "kaldi-decoder/python/csrc/faster-decoder.h"
#include "kaldi-decoder/python/csrc/lattice-faster-decoder.h"
#include "kaldi-decoder/python/csrc/lattice-simple-decoder.h"
#include "kaldi-decoder/python/csrc/simple-decoder.h"
#include "kaldi-decoder/python/csrc/transition-information.h"

namespace kaldi_decoder {

PYBIND11_MODULE(_kaldi_decoder, m) {
  m.doc() = "pybind11 binding of kaldi-decoder";
  PybindDecodableItf(&m);
  PybindTransitionInformation(&m);
  PybindDecodableMatrix(&m);
  PybindFasterDecoder(&m);
  PybindLatticeFasterDecoder(&m);
  PybindLatticeSimpleDecoder(&m);
  PybindSimpleDecoder(&m);
  PybindDecodableCtc(&m);
  PybindDecoderWrappers(&m);
}

}  // namespace kaldi_decoder
