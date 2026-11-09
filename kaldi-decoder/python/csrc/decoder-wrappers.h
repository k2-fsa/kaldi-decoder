// kaldi-decoder/python/csrc/decoder-wrappers.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef KALDI_DECODER_PYTHON_CSRC_DECODER_WRAPPERS_H_
#define KALDI_DECODER_PYTHON_CSRC_DECODER_WRAPPERS_H_

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

namespace kaldi_decoder {

void PybindDecoderWrappers(py::module *m);

}

#endif  // KALDI_DECODER_PYTHON_CSRC_DECODER_WRAPPERS_H_
