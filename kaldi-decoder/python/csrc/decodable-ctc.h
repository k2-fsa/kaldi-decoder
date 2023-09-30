// kaldi-decoder/python/csrc/decodable-ctc.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_DECODER_PYTHON_CSRC_DECODABLE_CTC_H_
#define KALDI_DECODER_PYTHON_CSRC_DECODABLE_CTC_H_

#include "kaldi-decoder/python/csrc/kaldi-decoder.h"

namespace kaldi_decoder {

void PybindDecodableCtc(py::module *m);

}

#endif  // KALDI_DECODER_PYTHON_CSRC_DECODABLE_CTC_H_
