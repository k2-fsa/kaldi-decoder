LatticeFasterDecoder
====================

kaldi's :kaldi:`LatticeFasterDecoder <src/decoder/lattice-faster-decoder.h>`
(see :kd:`kaldi-decoder/csrc/lattice-faster-decoder.h`): the standard
lattice decoder used by ``gmm-latgen-faster``.  Beam-pruned token-passing
that keeps alternative paths in a lattice, with separate ``beam`` (paths) and
``lattice_beam`` (lattice detail) controls.  This is the right default for
decoding with self-loops (HMM graphs).

Usage
-----

.. code-block:: python

   import kaldifst
   import kaldi_decoder as kd

   graph = kaldifst.StdVectorFst.read("HCLG.fst")   # keep it alive!
   decodable = MyDecodable()

   cfg = kd.LatticeFasterDecoderConfig(beam=13.0, lattice_beam=6.0,
                                       max_active=7000)
   decoder = kd.LatticeFasterDecoder(graph, cfg)

   ok = decoder.decode(decodable)             # one-shot ...
   # ... or incremental:
   decoder.init_decoding()
   decoder.advance_decoding(decodable)        # all available frames
   decoder.finalize_decoding()

   decoder.reached_final()
   ok, best = decoder.get_best_path()         # 1-best traceback
   ok, raw = decoder.get_raw_lattice()        # full lattice

Equivalent in kaldi
-------------------

.. code-block:: bash

   # this decoder is the engine of kaldi's gmm-latgen-faster
   gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 \
       --acoustic-scale=0.083 final.mdl HCLG.fst \
       "scp:feats.scp" "ark:|gzip -c > lat.1.gz"

Mapping
^^^^^^^

=========================================  ==========================================
kaldi                                      python API
=========================================  ==========================================
``LatticeFasterDecoderConfig::beam``       ``kd.LatticeFasterDecoderConfig(beam=...)``
``--beam``                                 ``cfg.beam``
``--max-active``                           ``cfg.max_active``
``--min-active``                           ``cfg.min_active``
``--lattice-beam``                         ``cfg.lattice_beam``
``--prune-interval``                       ``cfg.prune_interval``
``--beam-delta``                           ``cfg.beam_delta``
``--hash-ratio``                           ``cfg.hash_ratio``
``--determinize-lattice``                  ``cfg.determinize_lattice``
``Decode(decodable)``                      ``decode(decodable)``
``InitDecoding()/AdvanceDecoding()``       ``init_decoding()/advance_decoding()``
``GetBestPath(&lat, use_final)``           ``get_best_path(use_final_probs=True)``
``GetRawLattice(&lat, use_final)``         ``get_raw_lattice(use_final_probs=True)``
``ReachedFinal()``                         ``reached_final()``
``FinalRelativeCost()``                    ``final_relative_cost()``
``NumFramesDecoded()``                     ``num_frames_decoded()``
=========================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.LatticeFasterDecoderConfig
   :members:
   :special-members: __init__

.. autoclass:: kaldi_decoder.LatticeFasterDecoder
   :members:
   :special-members: __init__
