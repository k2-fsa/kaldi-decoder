// kaldi-decoder/python/csrc/decodable-matrix.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-decoder/python/csrc/decodable-matrix.h"

#include <memory>
#include <utility>

#include "kaldi-decoder/csrc/decodable-matrix.h"
#include "kaldi-decoder/csrc/transition-information.h"

namespace kaldi_decoder {

namespace {

// The kaldi classes borrow or own a Matrix; the pybind Eigen caster always
// converts numpy input to a TEMPORARY FloatMatrix.  To never leave the
// decodable pointing at a dead temporary, every ctor here materializes a
// heap FloatMatrix and transfers ownership:
//   - classes with kaldi's owning ctor use it (the class deletes the matrix),
//   - DecodableMatrixScaled only borrows, so wrap it in a subclass that owns
//     the copy (binding-side helper; the kaldi class is unchanged).
std::unique_ptr<FloatMatrix> MatrixFromNumpy(const FloatMatrix &converted) {
  return std::make_unique<FloatMatrix>(converted);
}

class OwnedDecodableMatrixScaled : public DecodableMatrixScaled {
 public:
  OwnedDecodableMatrixScaled(const FloatMatrix &likes, float scale)
      : DecodableMatrixScaled(likes_storage_, scale), likes_storage_(likes) {}

 private:
  FloatMatrix likes_storage_;
};

}  // namespace

void PybindDecodableMatrix(py::module *m) {
  {
    using PyClass = DecodableMatrixScaledMapped;
    auto cls =
        py::class_<PyClass, DecodableInterface>(
            *m, "DecodableMatrixScaledMapped",
            R"doc(Matrix-backed decodable with acoustic scale and tid->pdf mapping.

kaldi's `DecodableMatrixScaledMapped
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/decodable-matrix.h>`_.
The matrix must have ``tm.num_pdfs()`` columns; row ``t`` holds the per-pdf
log-likelihoods of frame ``t``.  The score of transition-id ``tid`` at frame
``t`` is ``scale * likes(t, tid_to_pdf[tid])``.

The numpy matrix is **copied** and owned by the decodable; it may be deleted
right after construction.

:param tm: Transition model providing the tid->pdf map.
:param likes: ``(num_frames, num_pdfs)`` float32 array.
:param scale: Acoustic scale (kaldi ``--acoustic-scale``).)doc")
            .def(py::init([](const TransitionInformation &tm,
                             const FloatMatrix &likes, float scale) {
                   auto mat = MatrixFromNumpy(likes);
                   return std::make_unique<PyClass>(tm, scale, mat.release());
                 }),
                 py::arg("tm"), py::arg("likes"), py::arg("scale"),
                 py::keep_alive<1, 2>(),
                 py::call_guard<py::gil_scoped_release>())
            .def("num_frames_ready", &PyClass::NumFramesReady,
                 R"doc(Number of frames (matrix rows).)doc")
            .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"),
                 R"doc(Whether ``frame`` is the last frame.)doc")
            .def("log_likelihood", &PyClass::LogLikelihood, py::arg("frame"),
                 py::arg("tid"),
                 R"doc(Scaled score of transition-id ``tid`` at ``frame``.

:param frame: Frame index, 0-based.
:param tid: Transition-id, 1-based.
:return: ``scale * likes(frame, tid_to_pdf[tid])``.)doc")
            .def("num_indices", &PyClass::NumIndices,
                 R"doc(Number of transition-ids (1-based).)doc");
    cls.attr("__module__") = "kaldi_decoder";
  }
  {
    using PyClass = DecodableMatrixMapped;
    auto cls =
        py::class_<PyClass, DecodableInterface>(
            *m, "DecodableMatrixMapped",
            R"doc(Matrix-backed decodable with a frame offset (chunked scores).

kaldi's `DecodableMatrixMapped
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/decodable-matrix.h>`_:
like :class:`DecodableMatrixScaledMapped` but without an acoustic scale and
with a frame offset -- row 0 of ``likes`` is frame ``frame_offset`` of the
utterance.  Useful when scores arrive in chunks.

The numpy matrix is **copied** and owned by the decodable.

:param tm: Transition model providing the tid->pdf map.
:param likes: ``(num_frames, num_pdfs)`` float32 array.
:param frame_offset: Frame index of row 0.)doc")
            .def(py::init([](const TransitionInformation &tm,
                             const FloatMatrix &likes, int32_t frame_offset) {
                   auto mat = MatrixFromNumpy(likes);
                   return std::make_unique<PyClass>(tm, mat.release(),
                                                    frame_offset);
                 }),
                 py::arg("tm"), py::arg("likes"), py::arg("frame_offset") = 0,
                 py::keep_alive<1, 2>(),
                 py::call_guard<py::gil_scoped_release>())
            .def("num_frames_ready", &PyClass::NumFramesReady,
                 R"doc(``frame_offset + likes.shape[0]``.)doc")
            .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"),
                 R"doc(Whether ``frame`` is the last frame.)doc")
            .def("log_likelihood", &PyClass::LogLikelihood, py::arg("frame"),
                 py::arg("tid"),
                 R"doc(Score of transition-id ``tid`` at ``frame`` (no scale).

:param frame: Frame index (``frame >= frame_offset``).
:param tid: Transition-id, 1-based.
:return: ``likes(frame - frame_offset, tid_to_pdf[tid])``.)doc")
            .def("num_indices", &PyClass::NumIndices,
                 R"doc(Number of transition-ids (1-based).)doc");
    cls.attr("__module__") = "kaldi_decoder";
  }
  {
    using PyClass = DecodableMatrixMappedOffset;
    auto cls =
        py::class_<PyClass, DecodableInterface>(
            *m, "DecodableMatrixMappedOffset",
            R"doc(Incrementally-fed matrix decodable for streaming/threads.

kaldi's `DecodableMatrixMappedOffset
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/decodable-matrix.h>`_:
call :meth:`accept_loglikes` whenever a new chunk of per-pdf scores is
available; old frames can be discarded to bound memory.  Querying a frame
below :meth:`first_available_frame` is an error.

:param tm: Transition model providing the tid->pdf map.)doc")
            .def(py::init<const TransitionInformation &>(), py::arg("tm"),
                 py::keep_alive<1, 2>(),
                 py::call_guard<py::gil_scoped_release>())
            .def("first_available_frame", &PyClass::FirstAvailableFrame,
                 R"doc(Frames below this index have been discarded.)doc")
            .def(
                "accept_loglikes",
                [](PyClass &self, const FloatMatrix &loglikes,
                   int32_t frames_to_discard) {
                  // kaldi semantics: AcceptLoglikes is destructive of its
                  // input; give it our private heap copy and free whatever
                  // remains (unique_ptr: freed even if AcceptLoglikes throws).
                  auto tmp = MatrixFromNumpy(loglikes);
                  self.AcceptLoglikes(tmp.get(), frames_to_discard);
                },
                py::arg("loglikes"), py::arg("frames_to_discard") = 0,
                R"doc(Append a chunk of per-pdf scores.

Destructive of the input (kaldi semantics): the input matrix may be
emptied.  ``frames_to_discard`` drops the OLDEST stored frames when the
chunk arrives.

:param loglikes: ``(T, num_pdfs)`` float32 array.
:param frames_to_discard: Number of oldest frames to drop.)doc",
                py::call_guard<py::gil_scoped_release>())
            .def("input_is_finished", &PyClass::InputIsFinished,
                 R"doc(Mark end of input; required before ``is_last_frame`` can
return ``True``.)doc")
            .def(
                "num_frames_ready", &PyClass::NumFramesReady,
                R"doc(Frames available so far (discarded frames are gone).)doc")
            .def(
                "is_last_frame", &PyClass::IsLastFrame, py::arg("frame"),
                R"doc(Whether ``frame`` is the last frame of a FINISHED input.)doc")
            .def("log_likelihood", &PyClass::LogLikelihood, py::arg("frame"),
                 py::arg("tid"),
                 R"doc(Score of transition-id ``tid`` at ``frame``.)doc")
            .def("num_indices", &PyClass::NumIndices,
                 R"doc(Number of transition-ids (1-based).)doc");
    cls.attr("__module__") = "kaldi_decoder";
  }
  {
    // Bound under the kaldi class name; the wrapper owns its matrix copy.
    using PyClass = OwnedDecodableMatrixScaled;
    using Base = DecodableMatrixScaled;
    auto cls =
        py::class_<PyClass, DecodableInterface>(
            *m, "DecodableMatrixScaled",
            R"doc(Plain matrix-backed decodable with an acoustic scale (no map).

kaldi's `DecodableMatrixScaled
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/decodable-matrix.h>`_:
columns ARE the indices (one-based): the score of ``index`` at frame ``t`` is
``scale * likes(t, index - 1)``.  Use when the caller already has per-index
scores (e.g. CTC-like graphs).

The numpy matrix is **copied** and owned by the decodable.

:param likes: ``(num_frames, num_indices)`` float32 array.
:param scale: Acoustic scale.)doc")
            .def(py::init<const FloatMatrix &, float>(), py::arg("likes"),
                 py::arg("scale"), py::call_guard<py::gil_scoped_release>())
            .def("num_frames_ready", &Base::NumFramesReady)
            .def("is_last_frame", &Base::IsLastFrame, py::arg("frame"))
            .def("log_likelihood", &Base::LogLikelihood, py::arg("frame"),
                 py::arg("index"))
            .def("num_indices", &Base::NumIndices);
  }
}

}  // namespace kaldi_decoder
