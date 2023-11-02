// kaldi-decoder/python/csrc/lattice-simple-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_DECODER_PYTHON_CSRC_LATTICE_SIMPLE_DECODER_H_
#define KALDI_DECODER_PYTHON_CSRC_LATTICE_SIMPLE_DECODER_H_

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

namespace kaldi_decoder {

void PybindLatticeSimpleDecoder(py::module *m);

}

#endif  // KALDI_DECODER_PYTHON_CSRC_LATTICE_SIMPLE_DECODER_H_
