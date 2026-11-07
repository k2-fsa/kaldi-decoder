// kaldi-decoder/python/csrc/fst-caster.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Zero-copy cross-module bridge for openfst types.
//
// Ported from ../kaldi-hmm-gmm/python/csrc/fst-caster.h (keep in sync).
//
// Each pybind11 module (kaldifst, kaldi-hmm-gmm, kaldi-decoder, ...) bundles
// its own hidden copy of openfst, so pybind11's std::type_index based
// cross-module type matching cannot work (per-module typeinfo never unifies).
// Instead of serializing FSTs across the boundary, we share them BY POINTER
// through kaldifst's explicit helpers (kaldifst.lib._kaldifst):
//
//   _vector_fst_addr(fst)        -> uint64_t   (raw C++ address)
//   _vector_fst_borrow(addr)     -> StdVectorFst  (non-owning wrapper)
//   _vector_fst_take(addr)       -> StdVectorFst  (owning wrapper)
//
// Contract (ABI convention): every module in the process must be built
// against the same openfst version and ABI-relevant flags (e.g.
// -fno-exceptions).  The object always stays owned by the module that
// created it; we never delete across module boundaries.
//
// Note: fst::VectorFst<fst::LatticeArc> / fst::ConstFst<fst::StdArc> rely on
// the kaldifst module's own pybind11 registration (it exposes the Lattice
// class hierarchy); this caster covers the StdArc VectorFst hot path.

#ifndef KALDI_DECODER_PYTHON_CSRC_FST_CASTER_H_
#define KALDI_DECODER_PYTHON_CSRC_FST_CASTER_H_

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <utility>

#include "fst/fstlib.h"

namespace pybind11::detail {

inline object kd_kaldifst_module() {
  // Intentionally leaked: a C++ static `object` would be destroyed during C++
  // static destruction, i.e. after python finalization, which aborts the
  // process at exit.  Hold a bare PyObject* that is never decref'd instead.
  static PyObject *mod = nullptr;
  if (mod == nullptr) {
    object obj = module_::import("kaldifst.lib").attr("_kaldifst");
    mod = obj.release().ptr();
  }
  return reinterpret_borrow<object>(mod);
}

template <>
struct type_caster<fst::VectorFst<fst::StdArc>> {
 public:
  static constexpr auto name = const_name("fst::VectorFst<fst::StdArc>");

  bool load(handle src, bool) {
    if (src.is_none()) return false;
    try {
      pybind11::gil_scoped_acquire gil;  // safe if already held
      uint64_t addr =
          kd_kaldifst_module().attr("_vector_fst_addr")(src).cast<uint64_t>();
      // NOLINTNEXTLINE(performance-no-int-to-ptr) -- pointer bridge by design
      held_ = reinterpret_cast<fst::VectorFst<fst::StdArc> *>(addr);
      return held_ != nullptr;
    } catch (const std::exception &) {
      return false;
    }
  }

  // Return path: hand a (moved) VectorFst to kaldifst as an owning wrapper.
  static handle cast(fst::VectorFst<fst::StdArc> &&src, return_value_policy,
                     handle) {
    auto *heap = new fst::VectorFst<fst::StdArc>(std::move(src));
    return wrap(heap, /*owning=*/true);
  }

  static handle cast(const fst::VectorFst<fst::StdArc> &src,
                     return_value_policy, handle) {
    auto *heap = new fst::VectorFst<fst::StdArc>(src);
    return wrap(heap, /*owning=*/true);
  }

  static handle cast(fst::VectorFst<fst::StdArc> *src, return_value_policy,
                     handle) {
    if (src == nullptr) return none().release();
    // SAFETY CONTRACT: kaldi-decoder never returns a *borrowed* FST (no
    // reference policy); a borrowed wrapper could outlive its owner
    // (use-after-free).  Pointer returns are always transferred to the
    // python side as owning.
    return wrap(src, /*owning=*/true);
  }

  operator fst::VectorFst<fst::StdArc> *() { return held_; }   // NOLINT
  operator fst::VectorFst<fst::StdArc> &() { return *held_; }  // NOLINT

  template <typename T_>
  using cast_op_type = movable_cast_op_type<T_>;

 private:
  static handle wrap(fst::VectorFst<fst::StdArc> *p, bool owning) {
    // The python API needs the GIL; acquire is reentrant so this is safe
    // regardless of caller context.
    pybind11::gil_scoped_acquire gil;
    uint64_t addr =
        reinterpret_cast<uint64_t>(p);  // NOLINT -- pointer bridge by design
    // Ownership always transfers to the python side (see the SAFETY
    // CONTRACT on the pointer-returning cast()).
    (void)owning;
    return kd_kaldifst_module().attr("_vector_fst_take")(addr).release();
  }

  fst::VectorFst<fst::StdArc> *held_ = nullptr;
};

// For decoder interfaces taking `const fst::Fst<fst::StdArc>&`.
template <>
struct type_caster<fst::Fst<fst::StdArc>> {
 public:
  static constexpr auto name = const_name("fst::Fst<fst::StdArc>");

  bool load(handle src, bool) {
    if (src.is_none()) return false;
    try {
      pybind11::gil_scoped_acquire gil;  // safe if already held
      uint64_t addr =
          kd_kaldifst_module().attr("_vector_fst_addr")(src).cast<uint64_t>();
      // NOLINTNEXTLINE(performance-no-int-to-ptr) -- pointer bridge by design
      held_ = reinterpret_cast<fst::VectorFst<fst::StdArc> *>(addr);
      return held_ != nullptr;
    } catch (const std::exception &) {
      return false;
    }
  }

  static handle cast(const fst::Fst<fst::StdArc> &src, return_value_policy,
                     handle) {
    // Best effort: copy the FST into a VectorFst and hand it over.
    pybind11::gil_scoped_acquire gil;  // safe if already held
    auto *heap = new fst::VectorFst<fst::StdArc>(src);
    return kd_kaldifst_module()
        .attr("_vector_fst_take")(reinterpret_cast<uint64_t>(heap))
        .release();
  }

  operator fst::Fst<fst::StdArc> *() {  // NOLINT
    return static_cast<fst::Fst<fst::StdArc> *>(held_);
  }
  operator fst::Fst<fst::StdArc> &() {  // NOLINT
    return *static_cast<fst::Fst<fst::StdArc> *>(held_);
  }
  operator const fst::Fst<fst::StdArc> *() const {  // NOLINT
    return static_cast<const fst::Fst<fst::StdArc> *>(held_);
  }
  operator const fst::Fst<fst::StdArc> &() const {  // NOLINT
    return *static_cast<const fst::Fst<fst::StdArc> *>(held_);
  }

  template <typename T_>
  using cast_op_type = movable_cast_op_type<T_>;

 private:
  fst::VectorFst<fst::StdArc> *held_ = nullptr;
};

}  // namespace pybind11::detail

#endif  // KALDI_DECODER_PYTHON_CSRC_FST_CASTER_H_
