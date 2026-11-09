# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`kaldi-decoder` (github.com/k2-fsa/kaldi-decoder) = kaldi's decoding machinery
(`src/decoder/`, parts of `src/lat/`, `src/itf/decodable-itf.h`, selected
`src/util/` support) reimplemented against kaldifst/openfst + Eigen, for reuse
by kaldi-hmm-gmm, sherpa-onnx, and standalone Python users.

- **C++ core** in `kaldi-decoder/csrc/` (namespace `kaldi_decoder`); file names
  and semantics mirror `../kaldi/src/{decoder,lat,itf,util}` — keep them aligned
  with kaldi. Provenance comment at the top of each ported file.
- **Python package** `kaldi_decoder` (pybind11), extension in
  `kaldi_decoder/lib/_kaldi_decoder.*` (self-contained: `kaldi-decoder-core` is
  always STATIC and linked in).
- **Zero AM/tree dependency** (architecture contract): the decoders only know
  `DecodableInterface` (kaldi's `itf/decodable-itf.h`). GMM/tree/training code
  lives in sibling project `../kaldi-hmm-gmm` (`khg::DecodableAmDiagGmmScaled`
  implements this repo's interface; `TransitionModel`, `ContextDependency`,
  `TrainingGraphCompiler` stay there). Never add dependencies on
  kaldi-hmm-gmm — it depends on THIS repo, not the other way round.

## Build & test

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2                      # keep it modest
ctest --output-on-failure     # gtest + python tests (<name>_py)
```

- Single test: `ctest -R <regexp>`; python tests run via
  `kaldi_decoder_add_py_test` (registered with `cmake -E env "PYTHONPATH=..."`
  — **never** ctest's `ENVIRONMENT` property: it silently drops path entries
  on Windows because `;` splits the CMake list).
- New gtest: add `xxx-test.cc` to `test_srcs` in `kaldi-decoder/csrc/CMakeLists.txt`.
- Python wheel: `python setup.py install` (cmake-based `cmake_extension`;
  version comes from `KALDI_DECODER_VERSION` in the top `CMakeLists.txt`).
- Style: `./scripts/check_style_cpplint.sh`.
- Building the reference kaldi (for parity work): `cd ../kaldi/src && make -j2`
  — **never more than 2 jobs**.
- **Zero CMake warnings** is a gate: `CMP0169 OLD` +
  `CMAKE_POLICY_DEFAULT_CMP0169 OLD` (FetchContent_Populate in
  `cmake/kaldifst.cmake` and inside fetched kaldifst/openfst) and
  `FetchContent_MakeAvailable` elsewhere. Don't reintroduce `Populate` calls
  outside kaldifst.cmake (it must populate before `add_subdirectory` to set
  `CMAKE_MODULE_PATH`).

## Architecture map

- `kaldi-decoder/csrc/` ↔ `../kaldi/src/`:
  - `decodable-itf.h` ← `itf/decodable-itf.h` (exact; Python trampoline
    `PyDecodableInterface` lets Python subclasses be the decodable)
  - `simple-decoder`, `faster-decoder`, `lattice-simple-decoder` ←
    `decoder/simple-decoder`, `decoder/faster-decoder`,
    `decoder/lattice-simple-decoder`
  - `lattice-faster-decoder` ← `decoder/lattice-faster-decoder`
  - `decoder-wrappers` ← `decoder/decoder-wrappers` (adapted: return-values
    instead of kaldi table-writers; best path via scalarized `fst::ShortestPath`
    with lm_scale/wip semantics matching `lattice-best-path`)
  - `hash-list*` ← `util/hash-list*`; `kaldi-math.h` ← `base/kaldi-math.h`;
    `stl-utils.h` ← `util/stl-utils.h`; `macros.h` = `KALDI_DECODER_LOGE()` logging
    (original, modeled on kaldi-native-fbank's `macros.h`); checks throw
    `std::runtime_error` directly (no assert macro)
  - `decodable-ctc` = original (matrix-backed decodable, no kaldi counterpart)
  - lattice types (`fst::LatticeArc`, `CompactLattice`, `ScaleLattice`) come
    from the **kaldifst** dependency — do not duplicate them here.
- `kaldi-decoder/python/csrc/` — one `Pybind*.cc` per C++ module, aggregated in
  `kaldi-decoder.cc` (`PYBIND11_MODULE _kaldi_decoder`); re-exported by
  `python/kaldi_decoder/__init__.py`.
- **FST interop with the `kaldifst` Python package** goes through
  `python/csrc/fst-caster.h` (zero-copy pointer bridge via
  `kaldifst.lib._kaldifst._vector_fst_addr/_borrow/_take`). Each pybind module
  bundles its own hidden openfst copy, so pybind11 type matching across modules
  cannot work; never delete an object across module boundaries (contract in
  the caster header). ABI contract: same openfst version and ABI flags in all
  modules of a process (keep `cmake/kaldifst.cmake` in sync with
  kaldi-hmm-gmm's = v1.9.0).
- Deps via `cmake/{kaldifst,eigen,pybind11,googletest}.cmake`
  (FetchContent + SHA256 + local-tarball fallback + `include_guard(GLOBAL)` +
  `if(TARGET ...) return()` so a parent project can supply them — kaldi-hmm-gmm
  injects its own `./cmake` modules when this repo is a subproject). Eigen
  version must equal kaldi-hmm-gmm's (5.0.1) — one Eigen per process.

## Standing working rules (user's instructions)

1. **kaldi is the only oracle.** Fill gaps from `../kaldi`: port the C++
   (keep names/semantics), then add pybind bindings. Never work around a C++
   gap in Python. Existing code here is untrusted too: source-diff each module
   against `../kaldi`, add tests where missing.
2. **Verify every step early — never batch tests.** Per module: (a) line-diff
   vs `../kaldi`; (b) gtest here (golden values, checked into `testdata/` with
   the generating kaldi command documented); (c) Python test re-verifying the
   binding on the same golden. All green before the next module.
3. **Parity policy**: ignore floating-point precision differences and
   random-number effects; report any remaining real behavioral difference to
   the user and let them decide. Do not silently align configs.
4. **No code before the user approves the plan** (plans live in the sibling
   repo's plan docs or here; user reviews and approves).
5. **Packaging contract**: python `.so` in `kaldi_decoder/lib/`;
   `kaldi-decoder-core` always STATIC (+PIC); deps only via `./cmake`
   (FetchContent pattern); test on all platforms (linux/macos x64+arm64,
   windows x64+arm64) with Windows-safe python test registration.
8. **C++ declaration style**: initialize with `Type x = 0;` — NEVER brace
   style (`Type x{0};` / `Type x{};`); one declarator per line — NEVER
   `Type x, y;` (write `Type x;` and `Type y;` on separate lines).
9. **if/else always use braces `{}`**, even for a single statement
   (`if (x) { return; }`, never `if (x) return;`).
10. **Blank lines for readability**: separate logical blocks with blank
   lines (between declarations groups, before/after `if`/`for` blocks,
   around `return`) — follow the style of `../sherpa-onnx/sherpa-onnx/csrc`.
6. **Memory safety**: python/C++ boundary objects cross via borrow/take with
   strict ownership (creator frees; `borrow` non-owning, `take` transfers).
   No leaks, no use-after-free; cover lifetimes in tests.  Decoder
   constructors use `py::keep_alive<1, 2>()` so a decoder holds the graph
   (temporaries are safe); never rely on the caller keeping it alive.
7. **GIL discipline**: the utterance-level bindings use
   `py::call_guard<py::gil_scoped_release>()`; anything that touches the
   Python C-API from those scopes must take the GIL first
   (`pybind11::gil_scoped_acquire`, reentrant).  The `PYBIND11_OVERRIDE*`
   macros do this automatically; hand-written trampoline methods and the
   fst-caster helpers must do it explicitly (see transition-information.cc
   and fst-caster.h).  `keep_alive` and other postcall policies run with the
   GIL held -- no extra work needed there.  One decoder/graph per thread
   while a call is in flight.
7. **No lhotse dependency, anywhere.**
8. **No `<filesystem>`/std::filesystem**; string path manipulation only.
9. **No git commit/push** — only `git add`; the user commits and pushes.
10. Code comments/docstrings in English.

## Local environment (this machine)

- Python env `~/py312` (Python 3.12): has `kaldifst`, `kaldi_native_io`,
  `numpy`; no torch. Build the extension for the env in use.
- Sibling repos: `../kaldi` (reference oracle), `../kaldi-hmm-gmm` (consumer,
  GMM/tree/training), `../kaldifst`, `../sherpa-onnx` (C++ consumer).
- kaldi binaries for parity live in `../kaldi/src/{bin,featbin,gmmbin}`.

## Current work (pick up here)

plan-3.md in `../kaldi-hmm-gmm` (approved) moves decoder code from
kaldi-hmm-gmm into this repo. Progress: Phase 0 done there (training-graph
weight fix + baselines); Phase 1 done here (deps unified kaldifst 1.9.0 /
eigen 5.0.1, `compile_commands.json` top-level-only, cmake zero-warning,
this file, fst-caster port, `python/tests/` with Windows-safe
`kaldi_decoder_add_py_test`, 6-platform CI). Decoder unit tests added
(simple/faster/lattice-simple gtests with hand goldens + matching python
tests; 8/8 green) — note kaldi semantics: `GetBestPath` returns the best
PARTIAL path and true even without final states. Now: Phase 1.5 (audit
existing code vs `../kaldi`), Phases 2-3 (complete `lattice-faster-decoder`,
port `decoder-wrappers` / `lat` options / `decodable-matrix` + bindings),
Phase 4 switchover in kaldi-hmm-gmm. Incremental gates and parity rules as
above.
