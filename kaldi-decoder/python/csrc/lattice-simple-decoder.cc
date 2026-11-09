// kaldi-decoder/python/csrc/lattice-simple-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/lattice-simple-decoder.h"

#include <utility>

#include "kaldi-decoder/csrc/lattice-simple-decoder.h"
#include "kaldi-decoder/python/csrc/fst-caster.h"

namespace kaldi_decoder {

static void PybindLatticeSimpleDecoderConfig(py::module *m) {
  using PyClass = LatticeSimpleDecoderConfig;

  auto cls =
      py::class_<PyClass>(
          *m, "LatticeSimpleDecoderConfig",
          R"doc(Options for :class:`LatticeSimpleDecoder` (kaldi's
`LatticeSimpleDecoderConfig
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/lattice-simple-decoder.h>`_).
All fields writable; defaults match kaldi's.)doc")
          .def(py::init<float, float, int32_t, bool, bool, float, float>(),
               py::arg("beam") = 16.0, py::arg("lattice_beam") = 10.0,
               py::arg("prune_interval") = 25,
               py::arg("determinize_lattice") = true,
               py::arg("prune_lattice") = true, py::arg("beam_ratio") = 0.9,
               py::arg("prune_scale") = 0.1)
          .def_readwrite("beam", &PyClass::beam)
          .def_readwrite("lattice_beam", &PyClass::lattice_beam)
          .def_readwrite("prune_interval", &PyClass::prune_interval)
          .def_readwrite("determinize_lattice", &PyClass::determinize_lattice)
          .def_readwrite("prune_lattice", &PyClass::prune_lattice)
          .def_readwrite("beam_ratio", &PyClass::beam_ratio)
          .def_readwrite("prune_scale", &PyClass::prune_scale)
          .def("__str__", &PyClass::ToString);
}

void PybindLatticeSimpleDecoder(py::module *m) {
  PybindLatticeSimpleDecoderConfig(m);

  using PyClass = LatticeSimpleDecoder;
  auto cls =
      py::class_<PyClass>(
          *m, "LatticeSimpleDecoder",
          R"doc(Lattice-generating Viterbi decoder (kaldi's `LatticeSimpleDecoder
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/lattice-simple-decoder.h>`_).

Like :class:`SimpleDecoder` but keeps alternative paths within
``lattice_beam`` as a lattice.  See :meth:`get_raw_lattice`.

**Lifetime:** the decoder holds a reference to the graph (py::keep_alive); the graph is released automatically when the decoder is destroyed.  The decodable is only needed during the decode calls.)doc")
          .def(py::init<const fst::Fst<fst::StdArc> &,
                        const LatticeSimpleDecoderConfig &>(),
               py::arg("fst"), py::arg("config"), py::keep_alive<1, 2>())
          .def(py::init<const fst::VectorFst<fst::StdArc> &,
                        const LatticeSimpleDecoderConfig &>(),
               py::arg("fst"), py::arg("config"), py::keep_alive<1, 2>())
          .def(py::init<const fst::ConstFst<fst::StdArc> &,
                        const LatticeSimpleDecoderConfig &>(),
               py::arg("fst"), py::arg("config"), py::keep_alive<1, 2>())
          .def("get_config", &PyClass::GetOptions,
               R"doc(Return the current config.)doc")
          .def("num_frames_decoded", &PyClass::NumFramesDecoded)
          .def("reached_final", &PyClass::ReachedFinal,
               R"doc(True if a final state was active on the last frame.)doc")
          .def("final_relative_cost", &PyClass::FinalRelativeCost,
               R"doc(Difference between best final and best last-frame cost;
``inf`` means no final state reached.)doc")
          .def(
              "decode",
              [](PyClass &self, DecodableInterface &decodable) -> bool {
                return self.Decode(&decodable);
              },
              py::arg("decodable"),
              R"doc(Decode to the end of the decodable's frames.)doc",
              py::call_guard<py::gil_scoped_release>())
          .def("init_decoding", &PyClass::InitDecoding,
               R"doc(Reset for a new utterance.)doc")
          .def("finalize_decoding", &PyClass::FinalizeDecoding,
               R"doc(Final pruning of forward links; improves raw-lattice
accuracy toward the end of the utterance.)doc",
               py::call_guard<py::gil_scoped_release>())
          .def(
              "get_best_path",
              [](PyClass &self, bool use_final_probs)
                  -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
                fst::VectorFst<fst::LatticeArc> fst;
                bool ok = self.GetBestPath(&fst, use_final_probs);
                return std::make_pair(ok, fst);
              },
              py::arg("use_final_probs") = true,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_raw_lattice",
              [](PyClass &self, bool use_final_probs)
                  -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
                fst::VectorFst<fst::LatticeArc> fst;
                bool ok = self.GetRawLattice(&fst, use_final_probs);
                return std::make_pair(ok, fst);
              },
              py::arg("use_final_probs") = true,
              py::call_guard<py::gil_scoped_release>());
}

}  // namespace kaldi_decoder
