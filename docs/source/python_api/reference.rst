Python API reference
====================

Every class and every field, with the kaldi counterpart of each item.

DecodableInterface
------------------

kaldi counterpart: :kaldi:`DecodableInterface <src/itf/decodable-itf.h>`.
Implement all four methods (indices are one-based):

* ``num_frames_ready() -> int`` -- how many frames have scores available.
* ``is_last_frame(frame) -> bool`` -- True for the last available frame.
* ``num_indices() -> int`` -- number of transition-ids (1-based).
* ``log_likelihood(frame, index) -> float`` -- acoustic log-likelihood of
  transition-id ``index`` at ``frame``; higher is better.

DecodableCtc
------------

Matrix-backed decodable for CTC-style scores.

* ``DecodableCtc(log_probs, offset=0)`` -- ``log_probs`` is a
  ``(num_frames, num_classes)`` float32 ``numpy.ndarray`` (copied).
  *kaldi*: spirit of ``DecodableMatrixScaled``.
* ``offset: int = 0`` -- frame index of row 0 (chunked streaming).

Decodable matrix classes
------------------------

kaldi counterpart: :kaldi:`decodable-matrix.h <src/decoder/decodable-matrix.h>`.
All four copy/own their numpy matrix.

* ``DecodableMatrixScaledMapped(tm, likes, scale)`` -- per-pdf scores
  ``likes`` (``num_frames x tm.num_pdfs()``), tid->pdf mapping via ``tm``,
  score = ``scale * likes[frame, tid_to_pdf[tid]]``.
* ``DecodableMatrixMapped(tm, likes, frame_offset=0)`` -- same mapping, no
  scale; row 0 is frame ``frame_offset``.
* ``DecodableMatrixMappedOffset(tm)`` -- incrementally fed:

  * ``accept_loglikes(loglikes, frames_to_discard=0)`` -- append a chunk,
    dropping ``frames_to_discard`` oldest frames (destructive of the input,
    as in kaldi);
  * ``input_is_finished()`` -- mark end of input;
  * ``first_available_frame()`` -- frames below this have been discarded.
* ``DecodableMatrixScaled(likes, scale)`` -- no mapping; index ``i`` is
  column ``i-1``.

TransitionInformation
---------------------

kaldi counterpart: :kaldi:`TransitionInformation <src/itf/transition-information.h>`.
Implement all pure methods; ``num_transition_ids()`` is derived.

* ``transition_ids_equivalent(a, b) -> bool``
* ``transition_id_is_start_of_phone(tid) -> bool``
* ``transition_id_to_phone(tid) -> int``
* ``is_final(tid) -> bool``
* ``is_self_loop(tid) -> bool``
* ``transition_id_to_pdf_array() -> list`` -- whole map, index 0 unused.
* ``num_pdfs() -> int``
* ``num_transition_ids() -> int`` -- ``len(tid_to_pdf) - 1``.

SimpleDecoder
-------------

kaldi counterpart: :kaldi:`SimpleDecoder <src/decoder/simple-decoder.h>`.

* ``SimpleDecoder(graph, beam)`` -- borrows ``graph``.
* ``decode(decodable) -> bool`` -- run to the end of available frames.
* ``reached_final() -> bool``
* ``get_best_path(use_final_probs=True) -> (bool, Lattice)``
* ``final_relative_cost() -> float`` -- ``inf`` means no final state reached.

FasterDecoder / FasterDecoderOptions
------------------------------------

kaldi counterpart: :kaldi:`FasterDecoder <src/decoder/faster-decoder.h>`.

* ``FasterDecoderOptions(beam=16.0, max_active=INT32_MAX, min_active=20,
  beam_delta=0.5, hash_ratio=2.0)`` -- all fields are writable.

  *kaldi*: ``--beam``, ``--max-active``, ``--min-active``, ``--beam-delta``,
  ``--hash-ratio``.

* ``FasterDecoder(graph, opts)`` -- borrows ``graph``.
* ``init_decoding()`` / ``advance_decoding(decodable, max_num_frames=-1)``
* ``decode(decodable) -> bool`` -- init + advance all frames.
* ``reached_final() -> bool`` / ``final_relative_cost() -> float``
* ``get_best_path(use_final_probs=True) -> (bool, Lattice)``
* ``num_frames_decoded() -> int`` / ``set_options(opts)`` / ``get_options()``

LatticeSimpleDecoder / LatticeSimpleDecoderConfig
-------------------------------------------------

kaldi counterpart: :kaldi:`LatticeSimpleDecoder <src/decoder/lattice-simple-decoder.h>`.

* ``LatticeSimpleDecoderConfig(beam=16.0, lattice_beam=10.0,
  prune_interval=25, determinize_lattice=True, prune_lattice=True,
  beam_ratio=0.9, prune_scale=0.1)`` -- all fields writable.
* ``LatticeSimpleDecoder(graph, config)`` -- borrows ``graph``.
* ``decode(decodable) -> bool``
* ``get_best_path(use_final_probs=True) -> (bool, Lattice)``
* ``get_raw_lattice(use_final_probs=True) -> (bool, Lattice)``
* ``reached_final() -> bool`` / ``final_relative_cost() -> float``
* ``init_decoding()`` / ``finalize_decoding()`` / ``num_frames_decoded()``
* ``get_config()``

LatticeFasterDecoder / LatticeFasterDecoderConfig
-------------------------------------------------

kaldi counterpart: :kaldi:`LatticeFasterDecoder <src/decoder/lattice-faster-decoder.h>`.

* ``LatticeFasterDecoderConfig(beam=16.0, max_active=INT32_MAX,
  min_active=200, lattice_beam=10.0, prune_interval=25,
  determinize_lattice=True, beam_delta=0.5, hash_ratio=2.0,
  prune_scale=0.1, memory_pool_tokens_block_size=256,
  memory_pool_links_block_size=256)`` -- all fields writable; ``check()``
  validates ranges (kaldi ``Check()``).
* ``LatticeFasterDecoder(graph, config)`` -- borrows ``graph``.
* ``decode(decodable) -> bool``
* ``init_decoding()`` / ``advance_decoding(decodable, max_num_frames=-1)`` /
  ``finalize_decoding()``
* ``get_best_path(use_final_probs=True) -> (bool, Lattice)``
* ``get_raw_lattice(use_final_probs=True) -> (bool, Lattice)``
* ``reached_final() -> bool`` / ``final_relative_cost() -> float``
* ``num_frames_decoded() -> int`` / ``set_options(cfg)`` / ``get_options()``

Decoder wrappers
----------------

kaldi counterpart: :kaldi:`decoder-wrappers.h <src/decoder/decoder-wrappers.h>`
and :kaldi:`lattice-best-path.cc <src/latbin/lattice-best-path.cc>`.

* ``AlignConfig(beam=200.0, retry_beam=0.0, careful=False)`` -- *kaldi*:
  ``--beam``, ``--retry-beam``, ``--careful`` (of ``gmm-align-compiled``).
* ``align_utterance_wrapper(config, utt, acoustic_scale, fst, decodable,
  num_done, num_error, num_retried, tot_like, frame_count) -> 7-tuple`` --
  returns ``(num_done, num_error, num_retried, tot_like, frame_count,
  alignment, words)``; the five counters are *inputs* that get incremented
  (kaldi's accumulating writers).  Mutates ``fst`` when
  ``config.careful`` is set (adds the careful loop).
* ``decode_utterance_lattice_faster(decoder, decodable, trans_model, utt,
  allow_partial, acoustic_scale) -> (ok, alignment, words, like, lat)`` --
  ``alignment``/``words`` are the decode-time best path (as kaldi writes its
  tables); ``lat`` is the raw lattice with acoustic costs divided by
  ``acoustic_scale`` (exactly as kaldi writes ``lat.gz``).
* ``decode_utterance_lattice_simple(...)`` -- same for
  ``LatticeSimpleDecoder``.
* ``lattice_best_path(lat, lm_scale=1.0, acoustic_scale=1.0) ->
  (ok, alignment, words, like)`` -- kaldi's ``lattice-best-path``: minimizes
  ``lm_scale * graph_cost + acoustic_scale * acoustic_cost``.
