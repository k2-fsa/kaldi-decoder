// kaldi-decoder/python/csrc/transition-information.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/transition-information.h"

#include "kaldi-decoder/csrc/transition-information.h"

namespace kaldi_decoder {

namespace {

// Trampoline so Python classes can implement TransitionInformation (kaldi's
// itf/transition-information.h).  NOTE on TransitionIdToPdfArray(): the C++
// interface returns a const reference that callers may STORE (e.g.
// decodable-matrix caches it); returning a reference to a temporary would
// dangle.  We therefore cache the vector inside the trampoline: the address
// is stable for the lifetime of this object and the contents are refreshed
// on every call.
class PyTransitionInformation : public TransitionInformation {
 public:
  using TransitionInformation::TransitionInformation;

  bool TransitionIdsEquivalent(int32_t trans_id1,
                               int32_t trans_id2) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, TransitionInformation,
                                "transition_ids_equivalent",
                                TransitionIdsEquivalent, trans_id1, trans_id2);
  }

  bool TransitionIdIsStartOfPhone(int32_t trans_id) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, TransitionInformation,
                                "transition_id_is_start_of_phone",
                                TransitionIdIsStartOfPhone, trans_id);
  }

  int32_t TransitionIdToPhone(int32_t trans_id) const override {
    PYBIND11_OVERRIDE_PURE_NAME(int32_t, TransitionInformation,
                                "transition_id_to_phone", TransitionIdToPhone,
                                trans_id);
  }

  bool IsFinal(int32_t trans_id) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, TransitionInformation, "is_final",
                                IsFinal, trans_id);
  }

  bool IsSelfLoop(int32_t trans_id) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, TransitionInformation, "is_self_loop",
                                IsSelfLoop, trans_id);
  }

  const std::vector<int32_t> &TransitionIdToPdfArray() const override {
    // The public bindings release the GIL (call_guard<gil_scoped_release>);
    // calling back into Python here must take it.  gil_scoped_acquire is
    // reentrant (PyGILState_Ensure), so this is safe when already held.
    pybind11::gil_scoped_acquire gil;
    pybind11::function impl =
        pybind11::get_override(this, "transition_id_to_pdf_array");
    if (!impl) {
      throw std::runtime_error(
          "TransitionInformation: transition_id_to_pdf_array() not "
          "implemented");
    }
    tid_to_pdf_cache_ = impl().cast<std::vector<int32_t>>();
    return tid_to_pdf_cache_;
  }

  int32_t NumPdfs() const override {
    PYBIND11_OVERRIDE_PURE_NAME(int32_t, TransitionInformation, "num_pdfs",
                                NumPdfs);
  }

 private:
  mutable std::vector<int32_t> tid_to_pdf_cache_;
};

}  // namespace

void PybindTransitionInformation(py::module *m) {
  using PyClass = TransitionInformation;
  auto cls = py::class_<PyClass, PyTransitionInformation>(
      *m, "TransitionInformation",
      R"doc(Transition-model interface used by the decode wrappers.

kaldi's `TransitionInformation
<https://github.com/kaldi-asr/kaldi/blob/master/src/itf/transition-information.h>`_:
provides the tid->pdf map and per-transition-id queries.  Implement it in
Python to use :func:`align_utterance_wrapper` / :func:`decode_utterance_lattice_faster`
with any transition model; ``kaldi_hmm_gmm.TransitionModel`` implements the
same interface in C++.

**Note:** :meth:`transition_id_to_pdf_array` must return the whole map on
every call (the trampoline caches it; that cached reference is what
decodables keep).)doc");
  cls.def(py::init<>(),
          R"doc(Implement the pure virtual methods in a subclass.)doc")
      .def("transition_ids_equivalent", &PyClass::TransitionIdsEquivalent,
           py::arg("trans_id1"), py::arg("trans_id2"),
           R"doc(Whether the two transition-ids share the same pdf and are
interchangeable in alignments.)doc")
      .def("transition_id_is_start_of_phone",
           &PyClass::TransitionIdIsStartOfPhone, py::arg("trans_id"),
           R"doc(Whether ``trans_id`` starts a new phone.)doc")
      .def("transition_id_to_phone", &PyClass::TransitionIdToPhone,
           py::arg("trans_id"),
           R"doc(Phone index of ``trans_id``.)doc")
      .def("is_final", &PyClass::IsFinal, py::arg("trans_id"),
           R"doc(Whether ``trans_id`` is a final (transition-to-final)
transition.)doc")
      .def("is_self_loop", &PyClass::IsSelfLoop, py::arg("trans_id"),
           R"doc(Whether ``trans_id`` is a self-loop transition.)doc")
      .def("transition_id_to_pdf_array", &PyClass::TransitionIdToPdfArray,
           R"doc(The whole tid->pdf map: index 0 unused, transition-ids are
1-based.)doc")
      .def("num_pdfs", &PyClass::NumPdfs,
           R"doc(Number of distinct pdfs.)doc")
      .def("num_transition_ids", &PyClass::NumTransitionIds,
           R"doc(Number of transition-ids (``len(tid_to_pdf) - 1``).)doc");
  cls.attr("__module__") = "kaldi_decoder";
}

}  // namespace kaldi_decoder
