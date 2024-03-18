// kaldi-decoder/python/csrc/decodable-ctc.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/decodable-ctc.h"

#include "kaldi-decoder/csrc/decodable-ctc.h"

namespace kaldi_decoder {

void PybindDecodableCtc(py::module *m) {
  using PyClass = DecodableCtc;
  py::class_<PyClass, DecodableInterface>(*m, "DecodableCtc")
      .def(py::init<const FloatMatrix &, int32_t>(), py::arg("feats"),
           py::arg("offset") = 0);
}

}  // namespace kaldi_decoder
