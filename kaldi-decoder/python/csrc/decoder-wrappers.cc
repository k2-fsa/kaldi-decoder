// kaldi-decoder/python/csrc/decoder-wrappers.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/decoder-wrappers.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kaldi-decoder/csrc/decoder-wrappers.h"
#include "kaldi-decoder/python/csrc/fst-caster.h"

namespace kaldi_decoder {

static void PybindAlignConfig(py::module *m) {
  using PyClass = AlignConfig;
  auto cls =
      py::class_<PyClass>(
          *m, "AlignConfig",
          R"doc(Alignment beam settings (kaldi's ``gmm-align-compiled`` flags).)doc")
          .def(py::init<float, float, bool>(), py::arg("beam") = 200.0,
               py::arg("retry_beam") = 0.0, py::arg("careful") = false)
          .def_readwrite(
              "beam", &PyClass::beam,
              R"doc(Wide alignment beam (kaldi ``--beam``, default 200).)doc")
          .def_readwrite(
              "retry_beam", &PyClass::retry_beam,
              R"doc(Retry beam on failure; 0 disables (kaldi ``--retry-beam``).)doc")
          .def_readwrite("careful", &PyClass::careful,
                         R"doc(Careful alignment: detects "blind alley" failures
(kaldi ``--careful``); MODIFIES the graph.)doc");
}

static void PybindAlignUtteranceWrapper(py::module *m) {
  m->def(
      "align_utterance_wrapper",
      [](const AlignConfig &config, const std::string &utt,
         float acoustic_scale, fst::VectorFst<fst::StdArc> *fst,
         DecodableInterface &decodable, int32_t num_done, int32_t num_error,
         int32_t num_retried, double tot_like, int64_t frame_count)
          -> std::tuple<int32_t, int32_t, int32_t, double, int64_t,
                        std::vector<int32_t>, std::vector<int32_t>> {
        std::vector<int32_t> alignment;
        std::vector<int32_t> words;
        AlignUtteranceWrapper(config, utt, acoustic_scale, fst, &decodable,
                              &num_done, &num_error, &num_retried, &tot_like,
                              &frame_count, &alignment, &words);

        return std::make_tuple(num_done, num_error, num_retried, tot_like,
                               frame_count, alignment, words);
      },
      py::arg("config"), py::arg("utt"), py::arg("acoustic_scale"),
      py::arg("fst"), py::arg("decodable"), py::arg("num_done"),
      py::arg("num_error"), py::arg("num_retried"), py::arg("tot_like"),
      py::arg("frame_count"), py::call_guard<py::gil_scoped_release>(),
      R"doc(Forced-align one utterance (kaldi's ``gmm-align-compiled`` core).

Returns ``(num_done, num_error, num_retried, tot_like, frame_count,
alignment, words)``.  The five counters are INPUTS that get incremented
(they mirror kaldi's accumulating table writers) -- thread them through a
loop over utterances.

:param config: Alignment beams (:class:`AlignConfig`).
:param utt: Utterance id (for log messages).
:param acoustic_scale: Acoustic scale used in the decode.
:param fst: Per-utterance training graph; **mutated** when
   ``config.careful`` is set (pass a copy to reuse it).
:param decodable: Scores (:class:`DecodableInterface`).
:return: The 7-tuple above; ``alignment`` is transition-ids,
   ``words`` the word ids.)doc");
}

static void PybindDecodeUtteranceLatticeSimple(py::module *m) {
  m->def(
      "decode_utterance_lattice_simple",
      [](LatticeSimpleDecoder &decoder,  // not const but is really an input.
         DecodableInterface &decodable,  // not const but is really an input.
         const TransitionInformation &trans_model, const std::string &utt,
         bool allow_partial, float acoustic_scale)
          -> std::tuple<bool, std::vector<int32_t>, std::vector<int32_t>,
                        double, fst::VectorFst<fst::LatticeArc>> {
        std::vector<int32_t> alignments;
        std::vector<int32_t> words;
        double log_like = 0;
        fst::VectorFst<fst::LatticeArc> lat;
        bool succeeded = DecodeUtteranceLatticeSimple(
            decoder, decodable, trans_model, utt, allow_partial, acoustic_scale,
            &alignments, &words, &log_like, &lat);
        return std::make_tuple(succeeded, alignments, words, log_like, lat);
      },
      py::arg("decoder"), py::arg("decodable"), py::arg("trans_model"),
      py::arg("utt"), py::arg("allow_partial"), py::arg("acoustic_scale"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Decode one utterance (kaldi's ``gmm-latgen-faster`` output block).

Returns ``(ok, alignment, words, like, lat)``.  ``alignment``/``words``
are the DECODE-TIME best path (what kaldi writes to its words/alignment
tables); ``lat`` is the raw lattice exactly as kaldi writes ``lat.gz``
(acoustic costs divided by ``acoustic_scale``).

:param decoder: A :class:`LatticeSimpleDecoder` (already holding the graph).
:param decodable: Scores (:class:`DecodableInterface`).
:param trans_model: Transition information.
:param utt: Utterance id (for log messages).
:param allow_partial: Output partial results if no final state was reached.
:param acoustic_scale: Acoustic scale used in the decode.
:return: See above.)doc");
}

static void PybindDecodeUtteranceLatticeFaster(py::module *m) {
  m->def(
      "decode_utterance_lattice_faster",
      [](LatticeFasterDecoder &decoder,  // not const but is really an input.
         DecodableInterface &decodable,  // not const but is really an input.
         const TransitionInformation &trans_model, const std::string &utt,
         bool allow_partial, float acoustic_scale)
          -> std::tuple<bool, std::vector<int32_t>, std::vector<int32_t>,
                        double, fst::VectorFst<fst::LatticeArc>> {
        std::vector<int32_t> alignments;
        std::vector<int32_t> words;
        double log_like = 0;
        fst::VectorFst<fst::LatticeArc> lat;
        bool succeeded = DecodeUtteranceLatticeFaster(
            decoder, decodable, trans_model, utt, allow_partial, acoustic_scale,
            &alignments, &words, &log_like, &lat);
        return std::make_tuple(succeeded, alignments, words, log_like, lat);
      },
      py::arg("decoder"), py::arg("decodable"), py::arg("trans_model"),
      py::arg("utt"), py::arg("allow_partial"), py::arg("acoustic_scale"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Decode one utterance (kaldi's ``gmm-latgen-faster`` output block).

Returns ``(ok, alignment, words, like, lat)``.  ``alignment``/``words``
are the DECODE-TIME best path (what kaldi writes to its words/alignment
tables); ``lat`` is the raw lattice exactly as kaldi writes ``lat.gz``
(acoustic costs divided by ``acoustic_scale``).

:param decoder: A :class:`LatticeSimpleDecoder` (already holding the graph).
:param decodable: Scores (:class:`DecodableInterface`).
:param trans_model: Transition information.
:param utt: Utterance id (for log messages).
:param allow_partial: Output partial results if no final state was reached.
:param acoustic_scale: Acoustic scale used in the decode.
:return: See above.)doc");
}

static void PybindLatticeBestPath(py::module *m) {
  m->def(
      "lattice_best_path",
      [](const fst::VectorFst<fst::LatticeArc> &lat, float lm_scale,
         float acoustic_scale) -> std::tuple<bool, std::vector<int32_t>,
                                             std::vector<int32_t>, double> {
        std::vector<int32_t> alignments;
        std::vector<int32_t> words;
        double log_like = 0;
        bool ok = LatticeBestPath(lat, lm_scale, acoustic_scale, &alignments,
                                  &words, &log_like);
        return std::make_tuple(ok, alignments, words, log_like);
      },
      py::arg("lat"), py::arg("lm_scale") = 1.0,
      py::arg("acoustic_scale") = 1.0, py::call_guard<py::gil_scoped_release>(),
      R"doc(1-best of a lattice under lm/acoustic scaling (kaldi's
`lattice-best-path <https://github.com/kaldi-asr/kaldi/blob/master/src/latbin/lattice-best-path.cc>`_).

Minimizes ``lm_scale * graph_cost + acoustic_scale * acoustic_cost``.
On the lattices returned by :func:`decode_utterance_lattice_faster` (acoustic
costs already divided by the decode acoustic scale), use
``acoustic_scale=1.0`` and sweep ``lm_scale`` for lmwt scoring.

:param lat: Input lattice (``kaldifst.Lattice``).
:param lm_scale: Weight of the graph cost (kaldi ``--lm-scale``).
:param acoustic_scale: Weight of the acoustic cost
   (kaldi ``--acoustic-scale``).
:return: ``(ok, alignment, words, like)``; ``ok`` is False when the
   lattice has no complete path (outputs empty).)doc");
}

void PybindDecoderWrappers(py::module *m) {
  PybindAlignConfig(m);
  m->attr("AlignConfig").attr("__module__") = "kaldi_decoder";
  PybindAlignUtteranceWrapper(m);
  PybindDecodeUtteranceLatticeSimple(m);
  PybindDecodeUtteranceLatticeFaster(m);
  PybindLatticeBestPath(m);
}

}  // namespace kaldi_decoder
