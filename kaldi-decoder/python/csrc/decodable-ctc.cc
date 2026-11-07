// kaldi-decoder/python/csrc/decodable-ctc.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/decodable-ctc.h"

#include "kaldi-decoder/csrc/decodable-ctc.h"

namespace kaldi_decoder {

void PybindDecodableCtc(py::module *m) {
  using PyClass = DecodableCtc;
  auto cls = py::class_<PyClass, DecodableInterface>(
      *m, "DecodableCtc",
      R"doc(Matrix-backed decodable for CTC-style scores.

A ``(num_frames, num_classes)`` matrix of log-probabilities becomes a
decodable; the matrix is copied.  Indices are **one-based**:
``log_likelihood(t, 1)`` reads column 0.

:param log_probs: Per-frame log-probs, ``numpy.ndarray`` (float32).
:param offset: Frame index of row 0 -- use a non-zero offset when the
   matrix is a later chunk of a streaming utterance.)doc");
  cls.def(py::init<const FloatMatrix &, int32_t>(), py::arg("log_probs"),
          py::arg("offset") = 0,
          R"doc(Construct from a log-probability matrix (copied).

:param log_probs: ``(num_frames, num_classes)`` float32 array.
:param offset: Frame index of row 0 (default 0).)doc")
      .def("num_frames_ready", &PyClass::NumFramesReady,
           R"doc(Number of frames: ``log_probs.shape[0] + offset``.)doc")
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"),
           R"doc(Whether ``frame`` is the last frame of the matrix.)doc")
      .def("log_likelihood", &PyClass::LogLikelihood, py::arg("frame"),
           py::arg("index"),
           R"doc(Log-prob at ``frame`` for class ``index - 1``.)doc")
      .def("num_indices", &PyClass::NumIndices,
           R"doc(Number of classes (columns); indices are 1-based.)doc");
  cls.attr("__module__") = "kaldi_decoder";
}

}  // namespace kaldi_decoder
