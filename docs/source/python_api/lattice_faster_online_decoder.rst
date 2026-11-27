LatticeFasterOnlineDecoder
==========================

kaldi's :kaldi:`LatticeFasterOnlineDecoder
<src/decoder/lattice-faster-online-decoder.h>` (see
:kd:`kaldi-decoder/csrc/lattice-faster-online-decoder.h`):
:class:`~kaldi_decoder.LatticeFasterDecoder` whose tokens also keep a
backpointer to their best predecessor.  The current best path can then be
read cheaply, without building the raw lattice, after every chunk of
audio.  kaldi's online2 code uses this for endpointing and for the
silence-weighted i-vector (``OnlineSilenceWeighting``), as vosk does.

Usage
-----

.. code-block:: python

   import kaldifst
   import kaldi_decoder as kd

   graph = kaldifst.StdVectorFst.read("HCLG.fst")   # keep it alive!
   cfg = kd.LatticeFasterDecoderConfig(beam=13.0, lattice_beam=6.0)
   decoder = kd.LatticeFasterOnlineDecoder(graph, cfg)

   decoder.init_decoding()
   for chunk in chunks:
       decodable.accept_loglikes(chunk)
       decoder.advance_decoding(decodable)
       # one transition-id per decoded frame (kaldi's
       # OnlineSilenceWeighting::ComputeCurrentTraceback)
       tids = decoder.get_best_path_transition_ids(use_final_probs=False)
   decoder.finalize_decoding()
   ok, best = decoder.get_best_path()

Mapping
^^^^^^^

===============================================  ==================================================
kaldi                                            python API
===============================================  ==================================================
``GetBestPath(&lat, use_final)``                 ``get_best_path(use_final_probs=True)``
``BestPathEnd()`` + ``TraceBackBestPath()`` loop ``get_best_path_transition_ids(use_final_probs)``
``GetRawLatticePruned(&lat, use_final, beam)``   ``get_raw_lattice_pruned(use_final_probs, beam)``
(everything else)                                same as :class:`~kaldi_decoder.LatticeFasterDecoder`
===============================================  ==================================================

API reference
-------------

.. autoclass:: kaldi_decoder.LatticeFasterOnlineDecoder
   :members:
   :special-members: __init__
