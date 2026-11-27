// kaldi-decoder/python/csrc/lattice-faster-online-decoder.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/lattice-faster-online-decoder.h"

#include <utility>
#include <vector>

#include "kaldi-decoder/csrc/lattice-faster-online-decoder.h"
#include "kaldi-decoder/python/csrc/fst-caster.h"

namespace kaldi_decoder {

void PybindLatticeFasterOnlineDecoder(py::module *m) {
  // LatticeFasterOnlineDecoder = LatticeFasterOnlineDecoderTpl<fst::StdFst>.
  // The graph is BORROWED (not owned): keep the python FST object alive for
  // the decoder's whole lifetime.
  using PyClass = LatticeFasterOnlineDecoder;
  auto cls =
      py::class_<PyClass>(
          *m, "LatticeFasterOnlineDecoder",
          R"doc(kaldi's `LatticeFasterOnlineDecoder
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/lattice-faster-online-decoder.h>`_:
:class:`LatticeFasterDecoder` whose tokens also keep a backpointer to their
best predecessor, so the current best path can be traced back cheaply
(no raw lattice) after every chunk -- what kaldi's online2 code
(endpointing, silence-weighted i-vectors) uses.

Same methods as :class:`LatticeFasterDecoder`, plus
:meth:`get_best_path_transition_ids` and :meth:`get_raw_lattice_pruned`.

**Lifetime:** the decoder holds a reference to the graph (py::keep_alive); the graph is released automatically when the decoder is destroyed.  The decodable is only needed during the decode calls.)doc")
          .def(py::init<const fst::Fst<fst::StdArc> &,
                        const LatticeFasterDecoderConfig &>(),
               py::arg("fst"), py::arg("config"), py::keep_alive<1, 2>())
          .def("set_options", &PyClass::SetOptions, py::arg("config"),
               R"doc(Replace the config between utterances.)doc")
          .def("get_options", &PyClass::GetOptions,
               R"doc(Return the current config.)doc")
          .def(
              "decode",
              [](PyClass &self, DecodableInterface &decodable) -> bool {
                return self.Decode(&decodable);
              },
              py::arg("decodable"),
              R"doc(Decode to the end of the decodable's frames; True if any
traceback is available.)doc",
              py::call_guard<py::gil_scoped_release>())
          .def("init_decoding", &PyClass::InitDecoding,
               R"doc(Start decoding; call before :meth:`advance_decoding`.)doc")
          .def(
              "advance_decoding",
              [](PyClass &self, DecodableInterface &decodable,
                 int32_t max_num_frames) {
                self.AdvanceDecoding(&decodable, max_num_frames);
              },
              py::arg("decodable"), py::arg("max_num_frames") = -1,
              R"doc(Decode up to ``max_num_frames`` more frames (all available
if -1).)doc",
              py::call_guard<py::gil_scoped_release>())
          .def(
              "finalize_decoding", &PyClass::FinalizeDecoding,
              R"doc(Final pruning; improves raw-lattice accuracy near the end.
After this, best-path calls require ``use_final_probs=True``.)doc",
              py::call_guard<py::gil_scoped_release>())
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
              R"doc(1-best traceback as a ``(ok, kaldifst.Lattice)`` pair,
read through the token backpointers (kaldi's online ``GetBestPath``).

:param use_final_probs: Include final-probs in the costs (kaldi default).)doc",
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_best_path_transition_ids",
              [](PyClass &self, bool use_final_probs) -> std::vector<int32_t> {
                return GetBestPathTransitionIds(self, use_final_probs);
              },
              py::arg("use_final_probs") = false,
              R"doc(Transition-ids of the current best path, one per decoded
frame (input-epsilon links skipped) -- the traceback kaldi's
``OnlineSilenceWeighting::ComputeCurrentTraceback`` reads.  Pass it to
``kaldi_hmm_gmm.OnlineSilenceWeighting.compute_current_traceback``.

:param use_final_probs: kaldi's online code uses ``False`` while decoding
   (must be ``True`` after :meth:`finalize_decoding`).
:return: ``list[int]`` of length :meth:`num_frames_decoded` (empty if no
   frame was decoded).)doc",
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
              R"doc(Raw lattice of near-best paths as a ``(ok, kaldifst.Lattice)``
pair; graph costs in Value1, acoustic costs in Value2.)doc",
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_raw_lattice_pruned",
              [](PyClass &self, bool use_final_probs, float beam)
                  -> std::pair<bool, fst::VectorFst<fst::LatticeArc>> {
                fst::VectorFst<fst::LatticeArc> fst;
                bool ok = self.GetRawLatticePruned(&fst, use_final_probs, beam);
                return std::make_pair(ok, fst);
              },
              py::arg("use_final_probs"), py::arg("beam"),
              R"doc(Like :meth:`get_raw_lattice` but keeps only tokens whose
extra cost is below ``beam`` (useful when ``beam < lattice_beam``).)doc",
              py::call_guard<py::gil_scoped_release>());
  cls.attr("__module__") = "kaldi_decoder";
}

}  // namespace kaldi_decoder
