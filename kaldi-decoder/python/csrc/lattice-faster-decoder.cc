// kaldi-decoder/python/csrc/lattice-faster-decoder.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/lattice-faster-decoder.h"

#include <limits>
#include <utility>

#include "kaldi-decoder/csrc/lattice-faster-decoder.h"
#include "kaldi-decoder/python/csrc/fst-caster.h"

namespace kaldi_decoder {

static void PybindLatticeFasterDecoderConfig(py::module *m) {
  using PyClass = LatticeFasterDecoderConfig;
  auto cls = py::class_<PyClass>(
      *m, "LatticeFasterDecoderConfig",
      R"doc(Options for :class:`LatticeFasterDecoder` (kaldi's
`LatticeFasterDecoderConfig
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/lattice-faster-decoder.h>`_).
All fields writable; defaults match kaldi's exactly.)doc")
      .def(py::init<float, int32_t, int32_t, float, int32_t, bool, float,
                    float, float, int32_t, int32_t>(),
           py::arg("beam") = 16.0,
           py::arg("max_active") = std::numeric_limits<int32_t>::max(),
           py::arg("min_active") = 200, py::arg("lattice_beam") = 10.0,
           py::arg("prune_interval") = 25,
           py::arg("determinize_lattice") = true, py::arg("beam_delta") = 0.5,
           py::arg("hash_ratio") = 2.0, py::arg("prune_scale") = 0.1,
           py::arg("memory_pool_tokens_block_size") = (1 << 8),
           py::arg("memory_pool_links_block_size") = (1 << 8))
      .def_readwrite("beam", &PyClass::beam,
                     R"doc(Decoder beam (kaldi ``--beam``): controls which
paths stay alive.)doc")
      .def_readwrite("max_active", &PyClass::max_active,
                     R"doc(Max active tokens per frame (kaldi ``--max-active``).)doc")
      .def_readwrite("min_active", &PyClass::min_active,
                     R"doc(Don't prune below this many tokens (kaldi ``--min-active``).)doc")
      .def_readwrite("lattice_beam", &PyClass::lattice_beam,
                     R"doc(Lattice beam (kaldi ``--lattice-beam``): how much
worse than the best path an alternative may be to stay in the lattice.)doc")
      .def_readwrite("prune_interval", &PyClass::prune_interval,
                     R"doc(Prune every this many frames (kaldi ``--prune-interval``).)doc")
      .def_readwrite("determinize_lattice", &PyClass::determinize_lattice,
                     R"doc(Whether the CALLER will determinize the output
lattice (kaldi ``--determinize-lattice``); not inspected by the decoder.)doc")
      .def_readwrite("beam_delta", &PyClass::beam_delta,
                     R"doc(Adaptive beam increment (kaldi ``--beam-delta``).)doc")
      .def_readwrite("hash_ratio", &PyClass::hash_ratio,
                     R"doc(Hash-table sizing ratio (kaldi ``--hash-ratio``).)doc")
      .def_readwrite("prune_scale", &PyClass::prune_scale,
                     R"doc(Internal token-pruning scale (rarely changed).)doc")
      .def_readwrite("memory_pool_tokens_block_size",
                     &PyClass::memory_pool_tokens_block_size,
                     R"doc(Token memory-pool block size.)doc")
      .def_readwrite("memory_pool_links_block_size",
                     &PyClass::memory_pool_links_block_size,
                     R"doc(Forward-link memory-pool block size.)doc")
      .def("check", &PyClass::Check,
           R"doc(Validate option ranges; raises on bad values (kaldi ``Check()``).)doc")
      .def("__str__", &PyClass::ToString);
}

void PybindLatticeFasterDecoder(py::module *m) {
  PybindLatticeFasterDecoderConfig(m);

  // LatticeFasterDecoder = LatticeFasterDecoderTpl<fst::StdFst, StdToken>.
  // The graph is BORROWED (not owned): keep the python FST object alive for
  // the decoder's whole lifetime.
  using PyClass = LatticeFasterDecoder;
  auto cls = py::class_<PyClass>(
      *m, "LatticeFasterDecoder",
      R"doc(The standard lattice decoder (kaldi's `LatticeFasterDecoder
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/lattice-faster-decoder.h>`_),
the engine of ``gmm-latgen-faster``.

Beam-pruned token passing that keeps near-best alternative paths (within
``lattice_beam``) as a lattice.  See :doc:`../design/decoders` for the
algorithm and :doc:`../usage/decoding` for a walkthrough.

**Lifetime:** the decoder holds a reference to the graph (py::keep_alive); the graph is released automatically when the decoder is destroyed.  The decodable is only needed during the decode calls.)doc")
      .def(py::init<const fst::Fst<fst::StdArc> &,
                    const LatticeFasterDecoderConfig &>(),
           py::arg("fst"), py::arg("config"),
           py::keep_alive<1, 2>())
      .def("set_options", &PyClass::SetOptions, py::arg("config"),
           R"doc(Replace the config between utterances.)doc")
      .def("get_options", &PyClass::GetOptions,
           R"doc(Return the current config.)doc")
      .def("decode",
           [](PyClass &self, DecodableInterface &decodable) -> bool {
             return self.Decode(&decodable);
           },
           py::arg("decodable"),
           R"doc(Decode to the end of the decodable's frames; True if any
traceback is available.)doc")
      .def("init_decoding", &PyClass::InitDecoding,
           R"doc(Start decoding; call before :meth:`advance_decoding`.)doc")
      .def("advance_decoding",
           [](PyClass &self, DecodableInterface &decodable,
              int32_t max_num_frames) {
             self.AdvanceDecoding(&decodable, max_num_frames);
           },
           py::arg("decodable"), py::arg("max_num_frames") = -1,
           R"doc(Decode up to ``max_num_frames`` more frames (all available
if -1).)doc")
      .def("finalize_decoding", &PyClass::FinalizeDecoding,
           R"doc(Final pruning; improves raw-lattice accuracy near the end.)doc")
      .def("reached_final", &PyClass::ReachedFinal,
           R"doc(True if a final state was active on the last frame.)doc")
      .def("num_frames_decoded", &PyClass::NumFramesDecoded,
           R"doc(Number of frames decoded so far.)doc")
      .def("final_relative_cost", &PyClass::FinalRelativeCost,
           R"doc(Difference between best final and best last-frame cost;
``inf`` means no final state reached.)doc")
      .def(
          "get_best_path",
          [](PyClass &self, bool use_final_probs)
              -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
            fst::VectorFst<fst::LatticeArc> fst;
            bool ok = self.GetBestPath(&fst, use_final_probs);
            return std::make_pair(ok, fst);
          },
          py::arg("use_final_probs") = true,
          R"doc(1-best traceback as a ``(ok, kaldifst.Lattice)`` pair.

:param use_final_probs: Include final-probs in the costs (kaldi default).)doc")
      .def(
          "get_raw_lattice",
          [](PyClass &self, bool use_final_probs)
              -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
            fst::VectorFst<fst::LatticeArc> fst;
            bool ok = self.GetRawLattice(&fst, use_final_probs);
            return std::make_pair(ok, fst);
          },
          py::arg("use_final_probs") = true,
          R"doc(Raw lattice of near-best paths as a ``(ok, kaldifst.Lattice)``
pair; graph costs in Value1, acoustic costs in Value2.)doc");
  cls.attr("__module__") = "kaldi_decoder";
}

}  // namespace kaldi_decoder
