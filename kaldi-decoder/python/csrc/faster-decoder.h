// kaldi-decoder/python/csrc/faster-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_DECODER_PYTHON_CSRC_FASTER_DECODER_H_
#define KALDI_DECODER_PYTHON_CSRC_FASTER_DECODER_H_

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

namespace kaldi_decoder {

void PybindFasterDecoder(py::module *m);

}

#endif  // KALDI_DECODER_PYTHON_CSRC_FASTER_DECODER_H_
