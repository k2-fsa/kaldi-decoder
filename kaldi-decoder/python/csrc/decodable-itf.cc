// kaldi-decoder/python/csrc/decodable-itf.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/decodable-itf.h"

#include "kaldi-decoder/csrc/decodable-itf.h"

namespace kaldi_decoder {

namespace {

//
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
// https://pybind11.readthedocs.io/en/stable/reference.html#inheritance
class PyDecodableInterface : public DecodableInterface {
 public:
  using DecodableInterface::DecodableInterface;

  float LogLikelihood(int32_t frame, int32_t index) override {
    PYBIND11_OVERRIDE_PURE_NAME(float, DecodableInterface, "log_likelihood",
                                LogLikelihood, frame, index);
  }

  bool IsLastFrame(int32_t frame) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, DecodableInterface, "is_last_frame",
                                IsLastFrame, frame);
  }

  int32_t NumFramesReady() const override {
    PYBIND11_OVERRIDE_NAME(int32_t, DecodableInterface, "num_frames_ready",
                           NumFramesReady);
  }

  int32_t NumIndices() const override {
    PYBIND11_OVERRIDE_PURE_NAME(int32_t, DecodableInterface, "num_indices",
                                NumIndices);
  }
};

}  // namespace

void PybindDecodableItf(py::module *m) {
  using PyClass = DecodableInterface;

  auto cls = py::class_<PyClass, PyDecodableInterface>(
      *m, "DecodableInterface",
      R"doc(Scoring interface every decoder consumes.

This is kaldi's `DecodableInterface
<https://github.com/kaldi-asr/kaldi/blob/master/src/itf/decodable-itf.h>`_.
It answers one question: *how likely is transition-id ``index`` at frame
``frame``?*  Implement it in Python to plug any acoustic model into the
decoders (a pybind11 trampoline keeps virtual dispatch on the Python side).

Indices are **one-based** (openfst convention, 0 is epsilon).

**Lifetime:** decoders borrow the decodable without owning it -- keep the
Python object alive for the whole decode.

.. seealso:: :class:`DecodableMatrixScaledMapped`
   For the fast matrix-backed implementation.)doc");
  cls.def(py::init<>(),
          R"doc(Construct a decodable; implement the four methods below.)doc")
      .def("log_likelihood", &PyClass::LogLikelihood, py::arg("frame"),
           py::arg("index"),
           R"doc(Acoustic log-likelihood of transition-id ``index`` at frame
``frame`` (higher is better).

:param frame: Frame index, 0-based, ``frame < num_frames_ready()``.
:param index: Transition-id, 1-based, ``index <= num_indices()``.
:return: The log-likelihood.)doc")
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"),
           R"doc(Whether ``frame`` is the last frame with scores available.

For streaming decodables this is only true once the input is finished.)doc")
      .def("num_frames_ready", &PyClass::NumFramesReady,
           R"doc(Number of frames with scores available; may grow over time.)doc")
      .def("num_indices", &PyClass::NumIndices,
           R"doc(Number of valid indices (transition-ids); indices are
1-based.)doc");
  cls.attr("__module__") = "kaldi_decoder";
}

}  // namespace kaldi_decoder
