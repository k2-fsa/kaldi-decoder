// kaldi-decoder/python/csrc/decodable-itf.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_DECODER_PYTHON_CSRC_DECODABLE_ITF_H_
#define KALDI_DECODER_PYTHON_CSRC_DECODABLE_ITF_H_

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

namespace kaldi_decoder {

void PybindDecodableItf(py::module *m);

}

#endif  // KALDI_DECODER_PYTHON_CSRC_DECODABLE_ITF_H_
