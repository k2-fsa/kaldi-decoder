// kaldi-decoder/python/csrc/simple-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/simple-decoder.h"

#include <utility>

#include "kaldi-decoder/csrc/simple-decoder.h"
#include "kaldi-decoder/python/csrc/fst-caster.h"

namespace kaldi_decoder {

void PybindSimpleDecoder(py::module *m) {
  using PyClass = SimpleDecoder;
  auto cls = py::class_<PyClass>(
      *m, "SimpleDecoder",
      R"doc(Simple Viterbi decoder (kaldi's `SimpleDecoder
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/simple-decoder.h>`_).

Keeps one token per FST state; good for small graphs or as a reference.

**Lifetime:** the decoder holds a reference to the graph (py::keep_alive); the graph is released automatically when the decoder is destroyed.  The decodable is only needed during the decode calls.)doc")
      .def(py::init<const fst::Fst<fst::StdArc> &, float>(), py::arg("fst"),
           py::arg("beam"), py::keep_alive<1, 2>())
      .def(py::init<const fst::VectorFst<fst::StdArc> &, float>(),
           py::arg("fst"), py::arg("beam"),
           py::keep_alive<1, 2>())
      .def(py::init<const fst::ConstFst<fst::StdArc> &, float>(),
           py::arg("fst"), py::arg("beam"),
           py::keep_alive<1, 2>())
      .def("decode",
           [](PyClass &self, DecodableInterface &decodable) -> bool {
             return self.Decode(&decodable);
           },
           py::arg("decodable"),
           R"doc(Decode to the end of the decodable's frames; True if any
tokens survived.)doc")
      .def("reached_final", &PyClass::ReachedFinal,
           R"doc(True if a final state was active on the last frame.)doc")
      .def(
          "get_best_path",
          [](PyClass &self, bool use_final_probs)
              -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
            fst::VectorFst<fst::LatticeArc> fst;
            bool ok = self.GetBestPath(&fst, use_final_probs);
            return std::make_pair(ok, fst);
          },
          py::arg("use_final_probs") = true)
      .def("final_relative_cost", &PyClass::FinalRelativeCost,
           R"doc(Difference between the best final cost and the best cost on
the last frame; ``inf`` means no final state was reached.)doc")
      .def("init_decoding", &PyClass::InitDecoding,
           R"doc(Reset for a new utterance.)doc")
      .def("advance_decoding",
           [](PyClass &self, DecodableInterface &decodable,
              int32_t max_num_frames) {
             self.AdvanceDecoding(&decodable, max_num_frames);
           },
           py::arg("decodable"), py::arg("max_num_frames") = -1)
      .def("num_frames_decoded", &PyClass::NumFramesDecoded);
}

}  // namespace kaldi_decoder
