// kaldi-decoder/python/csrc/lattice-faster-online-decoder.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef KALDI_DECODER_PYTHON_CSRC_LATTICE_FASTER_ONLINE_DECODER_H_
#define KALDI_DECODER_PYTHON_CSRC_LATTICE_FASTER_ONLINE_DECODER_H_

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

namespace kaldi_decoder {

void PybindLatticeFasterOnlineDecoder(py::module *m);

}

#endif  // KALDI_DECODER_PYTHON_CSRC_LATTICE_FASTER_ONLINE_DECODER_H_
