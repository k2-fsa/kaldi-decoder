Decoder wrappers
================

Utterance-level alignment and decoding helpers, adapted from kaldi's
:kaldi:`decoder-wrappers.h <src/decoder/decoder-wrappers.h>` (see
:kd:`kaldi-decoder/csrc/decoder-wrappers.h`).  Where kaldi's binaries write
tables, these return plain Python lists:

* ``align_utterance_wrapper`` -- forced alignment of a known word sequence
  (the engine of kaldi's ``gmm-align-compiled``);
* ``decode_utterance_lattice_faster`` / ``decode_utterance_lattice_simple`` --
  one-utterance decoding (the engine of ``gmm-latgen-*``);
* ``lattice_best_path`` -- kaldi's ``lattice-best-path`` core: 1-best of a
  lattice with lm-scale / acoustic-scale / word-ins-penalty scoring.

Usage
-----

.. code-block:: python

   import kaldifst
   import kaldi_decoder as kd

   # ---- forced alignment (== gmm-align-compiled) ----
   graph = kaldifst.StdVectorFst.read("train_graph.fst")   # keep alive
   decodable = MyDecodable()
   (num_done, num_error, num_retried, tot_like, frame_count,
    alignment, words) = kd.align_utterance_wrapper(
        config=kd.AlignConfig(beam=200.0, retry_beam=0.0, careful=False),
        utt="utt1", acoustic_scale=0.1, fst=graph, decodable=decodable,
        num_done=0, num_error=0, num_retried=0, tot_like=0.0, frame_count=0)

   # ---- decoding (== gmm-latgen-faster) ----
   decoder = kd.LatticeFasterDecoder(graph, kd.LatticeFasterDecoderConfig())
   ok, alignment, words, like, lat = kd.decode_utterance_lattice_faster(
       decoder=decoder, decodable=decodable, trans_model=tm,
       utt="utt1", allow_partial=False, acoustic_scale=0.1)

   # ---- scoring the lattice (== lattice-best-path) ----
   # `lat` is written like kaldi writes it: acoustic costs / acoustic_scale.
   # Sweep lm weights / word insertion penalties like local/score.sh does:
   for lmwt in (0.5, 0.75, 1.0):
       ok, ali, words, like = kd.lattice_best_path(
           lat, lm_scale=lmwt, acoustic_scale=1.0)

**Lifetime:** the graph is borrowed (keep it alive); the returned ``lat`` is
an owned copy.

Equivalent in kaldi
-------------------

.. code-block:: bash

   # kaldi: forced alignment
   gmm-align-compiled --transition-scale=1.0 --acoustic-scale=0.1 \
       --self-loop-scale=0.1 --beam=200 --retry-beam=0 final.mdl \
       "ark:graphs.fsts" "scp:feats.scp" "ark:|gzip -c > ali.gz"

   # kaldi: decoding
   gmm-latgen-faster --acoustic-scale=0.1 final.mdl HCLG.fst \
       "scp:feats.scp" "ark:|gzip -c > lat.gz"

   # kaldi: lattice scoring (kaldi's local/score.sh loop)
   lattice-best-path --lm-scale=0.75 --word-ins-penalty=0 \
       "ark:gunzip -c lat.gz|" ark:tra.ark

Mapping
^^^^^^^

===========================================  ==========================================
kaldi                                        python API
===========================================  ==========================================
``gmm-align-compiled``                       ``kd.align_utterance_wrapper(...)``
``AlignConfig`` (beam/retry-beam/careful)    ``kd.AlignConfig(beam, retry_beam, careful)``
``gmm-latgen-faster``                        ``kd.decode_utterance_lattice_faster(...)``
``gmm-latgen-simple``                        ``kd.decode_utterance_lattice_simple(...)``
``lattice-best-path``                        ``kd.lattice_best_path(lat, lm_scale, acoustic_scale)``
``--lm-scale``                               ``lm_scale``
``--acoustic-scale``                         ``acoustic_scale``
words/alignment tables                       ``words`` / ``alignment`` (``list``)
``lat.gz``                                   ``lat`` (``kaldifst.Lattice``)
===========================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.AlignConfig
   :members:
   :special-members: __init__

.. autofunction:: kaldi_decoder.align_utterance_wrapper

.. autofunction:: kaldi_decoder.decode_utterance_lattice_faster

.. autofunction:: kaldi_decoder.decode_utterance_lattice_simple

.. autofunction:: kaldi_decoder.lattice_best_path
