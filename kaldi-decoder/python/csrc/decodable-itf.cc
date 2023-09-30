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

  py::class_<PyClass, PyDecodableInterface>(*m, "DecodableInterface")
      .def(py::init<>())
      .def("log_likelihood", &PyClass::LogLikelihood, py::arg("frame"),
           py::arg("index"))
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"))
      .def("num_frames_ready", &PyClass::NumFramesReady)
      .def("num_indices", &PyClass::NumIndices);
}

}  // namespace kaldi_decoder
