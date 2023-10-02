// kaldi-decoder/python/csrc/faster-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/faster-decoder.h"

#include <limits>
#include <utility>

#include "kaldi-decoder/csrc/faster-decoder.h"

namespace kaldi_decoder {

static void PybindFasterDecoderOptions(py::module *m) {
  using PyClass = FasterDecoderOptions;
  py::class_<PyClass>(*m, "FasterDecoderOptions")
      .def(py::init<float, int32_t, int32_t, float, float>(),
           py::arg("beam") = 16.0,
           py::arg("max_active") = std::numeric_limits<int32_t>::max(),
           py::arg("min_active") = 20, py::arg("beam_delta") = 0.5,
           py::arg("hash_ratio") = 2.0)
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("max_active", &PyClass::max_active)
      .def_readwrite("min_active", &PyClass::min_active)
      .def_readwrite("beam_delta", &PyClass::beam_delta)
      .def_readwrite("hash_ratio", &PyClass::hash_ratio)
      .def("__str__", &PyClass::ToString);
}

void PybindFasterDecoder(py::module *m) {
  PybindFasterDecoderOptions(m);
  using PyClass = FasterDecoder;
  py::class_<PyClass>(*m, "FasterDecoder")
      .def(py::init<const fst::Fst<fst::StdArc> &,
                    const FasterDecoderOptions &>(),
           py::arg("fst"), py::arg("config"))
      .def(py::init<const fst::VectorFst<fst::StdArc> &,
                    const FasterDecoderOptions &>(),
           py::arg("fst"), py::arg("config"))
      .def(py::init<const fst::ConstFst<fst::StdArc> &,
                    const FasterDecoderOptions &>(),
           py::arg("fst"), py::arg("config"))
      .def("set_options", &PyClass::SetOptions, py::arg("config"))
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
      .def("init_decoding", &PyClass::InitDecoding)
      .def("advanced_decoding", &PyClass::AdvanceDecoding, py::arg("decodable"),
           py::arg("max_num_frames") = -1)
      .def("num_frames_decoded", &PyClass::NumFramesDecoded);
}

}  // namespace kaldi_decoder
