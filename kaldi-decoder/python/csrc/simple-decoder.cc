// kaldi-decoder/python/csrc/simple-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/simple-decoder.h"

#include <utility>

#include "kaldi-decoder/csrc/simple-decoder.h"

namespace kaldi_decoder {

void PybindSimpleDecoder(py::module *m) {
  using PyClass = SimpleDecoder;
  py::class_<PyClass>(*m, "SimpleDecoder")
      .def(py::init<const fst::Fst<fst::StdArc> &, float>(), py::arg("fst"),
           py::arg("beam"))
      .def(py::init<const fst::VectorFst<fst::StdArc> &, float>(),
           py::arg("fst"), py::arg("beam"))
      .def(py::init<const fst::ConstFst<fst::StdArc> &, float>(),
           py::arg("fst"), py::arg("beam"))
      .def("decode", &PyClass::Decode, py::arg("decodable"))
      .def("reached_final", &PyClass::ReachedFinal)
      .def(
          "get_best_path",
          [](PyClass &self, bool use_final_probs)
              -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
            fst::VectorFst<fst::LatticeArc> fst;
            bool ok = self.GetBestPath(&fst, use_final_probs);
            return std::make_pair(ok, fst);
          },
          py::arg("use_final_probs") = true)
      .def("final_relative_cost", &PyClass::FinalRelativeCost)
      .def("init_decoding", &PyClass::InitDecoding)
      .def("advance_decoding", &PyClass::AdvanceDecoding, py::arg("decodable"),
           py::arg("max_num_frames") = -1)
      .def("num_frames_decoded", &PyClass::NumFramesDecoded);
}

}  // namespace kaldi_decoder
