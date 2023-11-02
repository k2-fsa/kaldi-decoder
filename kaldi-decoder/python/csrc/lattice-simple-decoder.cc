// kaldi-decoder/python/csrc/lattice-simple-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/lattice-simple-decoder.h"

#include "kaldi-decoder/csrc/lattice-simple-decoder.h"

namespace kaldi_decoder {

static void PybindLatticeSimpleDecoderConfig(py::module *m) {
  using PyClass = LatticeSimpleDecoderConfig;

  py::class_<PyClass>(*m, "LatticeSimpleDecoderConfig")
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
  py::class_<PyClass>(*m, "LatticeSimpleDecoder")
      .def(py::init<const fst::Fst<fst::StdArc> &,
                    const LatticeSimpleDecoderConfig &>(),
           py::arg("fst"), py::arg("config"))
      .def(py::init<const fst::VectorFst<fst::StdArc> &,
                    const LatticeSimpleDecoderConfig &>(),
           py::arg("fst"), py::arg("config"))
      .def(py::init<const fst::ConstFst<fst::StdArc> &,
                    const LatticeSimpleDecoderConfig &>(),
           py::arg("fst"), py::arg("config"))
      .def("get_config", &PyClass::GetOptions)
      .def("num_frames_decoded", &PyClass::NumFramesDecoded)
      .def("final_relative_cost", &PyClass::FinalRelativeCost)
      .def("decode", &PyClass::Decode, py::arg("decodable"))
      .def("init_decoding", &PyClass::InitDecoding)
      .def("finalize_decoding", &PyClass::FinalizeDecoding)
      .def(
          "get_best_path",
          [](PyClass &self, bool use_final_probs)
              -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
            fst::VectorFst<fst::LatticeArc> fst;
            bool ok = self.GetBestPath(&fst, use_final_probs);
            return std::make_pair(ok, fst);
          },
          py::arg("use_final_probs") = true)
      .def(
          "get_raw_lattice",
          [](PyClass &self, bool use_final_probs)
              -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
            fst::VectorFst<fst::LatticeArc> fst;
            bool ok = self.GetRawLattice(&fst, use_final_probs);
            return std::make_pair(ok, fst);
          },
          py::arg("use_final_probs") = true);
}

}  // namespace kaldi_decoder
