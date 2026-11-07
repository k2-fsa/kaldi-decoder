// kaldi-decoder/python/csrc/faster-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/faster-decoder.h"

#include <limits>
#include <utility>

#include "kaldi-decoder/csrc/faster-decoder.h"
#include "kaldi-decoder/python/csrc/fst-caster.h"

namespace kaldi_decoder {

static void PybindFasterDecoderOptions(py::module *m) {
  using PyClass = FasterDecoderOptions;
  auto cls = py::class_<PyClass>(
      *m, "FasterDecoderOptions",
      R"doc(Options for :class:`FasterDecoder` (kaldi's `FasterDecoderOptions
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/faster-decoder.h>`_).

All fields are writable; the defaults match kaldi's.)doc")
      .def(py::init<float, int32_t, int32_t, float, float>(),
           py::arg("beam") = 16.0,
           py::arg("max_active") = std::numeric_limits<int32_t>::max(),
           py::arg("min_active") = 20, py::arg("beam_delta") = 0.5,
           py::arg("hash_ratio") = 2.0)
      .def_readwrite("beam", &PyClass::beam,
                     R"doc(Decoding beam; larger is slower but more accurate
(kaldi ``--beam``).)doc")
      .def_readwrite("max_active", &PyClass::max_active,
                     R"doc(Maximum active tokens per frame (kaldi ``--max-active``).)doc")
      .def_readwrite("min_active", &PyClass::min_active,
                     R"doc(Don't prune while fewer tokens than this are
active (kaldi ``--min-active``).)doc")
      .def_readwrite("beam_delta", &PyClass::beam_delta,
                     R"doc(Increment used in the adaptive beam
(kaldi ``--beam-delta``).)doc")
      .def_readwrite("hash_ratio", &PyClass::hash_ratio,
                     R"doc(Hash-table sizing ratio (kaldi ``--hash-ratio``).)doc")
      .def("__str__", &PyClass::ToString);
}

void PybindFasterDecoder(py::module *m) {
  PybindFasterDecoderOptions(m);
  using PyClass = FasterDecoder;
  auto cls = py::class_<PyClass>(
      *m, "FasterDecoder",
      R"doc(Beam-pruned Viterbi decoder (kaldi's `FasterDecoder
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/faster-decoder.h>`_).

For FSTs without self-loops and without the need for a lattice; the fastest
decoder.  Use :class:`LatticeFasterDecoder` for general HMM graphs.

**Lifetime:** the decoder holds a reference to the graph (py::keep_alive); the graph is released automatically when the decoder is destroyed.  The decodable is only needed during the decode calls.)doc")
      .def(py::init<const fst::Fst<fst::StdArc> &,
                    const FasterDecoderOptions &>(),
           py::arg("fst"), py::arg("config"),
           py::keep_alive<1, 2>())
      .def(py::init<const fst::VectorFst<fst::StdArc> &,
                    const FasterDecoderOptions &>(),
           py::arg("fst"), py::arg("config"),
           py::keep_alive<1, 2>())
      .def(py::init<const fst::ConstFst<fst::StdArc> &,
                    const FasterDecoderOptions &>(),
           py::arg("fst"), py::arg("config"),
           py::keep_alive<1, 2>())
      .def("set_options", &PyClass::SetOptions, py::arg("config"))
      .def("decode", [](PyClass &self, DecodableInterface &decodable) {
           self.Decode(&decodable);
         }, py::arg("decodable"),
           R"doc(Decode to the end of the decodable's frames.  (Returns
nothing -- kaldi's FasterDecoder::Decode is void; query
:meth:`reached_final` / :meth:`get_best_path` afterwards.)doc")
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
      .def("init_decoding", &PyClass::InitDecoding,
           R"doc(Start decoding; call before :meth:`advance_decoding`.)doc")
      .def("advance_decoding",
           [](PyClass &self, DecodableInterface &decodable,
              int32_t max_num_frames) {
             self.AdvanceDecoding(&decodable, max_num_frames);
           },
           py::arg("decodable"), py::arg("max_num_frames") = -1)
      .def("num_frames_decoded", &PyClass::NumFramesDecoded);
}

}  // namespace kaldi_decoder
