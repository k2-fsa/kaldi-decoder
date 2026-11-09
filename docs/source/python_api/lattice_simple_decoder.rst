LatticeSimpleDecoder
====================

kaldi's :kaldi:`LatticeSimpleDecoder <src/decoder/lattice-simple-decoder.h>`
(see :kd:`kaldi-decoder/csrc/lattice-simple-decoder.h`): like SimpleDecoder
but keeps a lattice (alternative paths with their acoustic/graph costs)
instead of just the 1-best.  Use it when you need ``get_raw_lattice()`` on a
small graph.

Usage
-----

.. code-block:: python

   import kaldifst
   import kaldi_decoder as kd

   graph = kaldifst.StdVectorFst.read("HCLG.fst")   # keep it alive!
   decodable = MyDecodable()

   cfg = kd.LatticeSimpleDecoderConfig(beam=16.0, lattice_beam=10.0)
   decoder = kd.LatticeSimpleDecoder(graph, cfg)
   ok = decoder.decode(decodable)
   decoder.reached_final()
   ok, best = decoder.get_best_path()      # 1-best as a kaldifst Lattice
   ok, raw = decoder.get_raw_lattice()     # alternatives within lattice_beam

   # or drive it incrementally
   decoder.init_decoding()
   decoder.finalize_decoding()

Equivalent in kaldi
-------------------

.. code-block:: bash

   # kaldi's lattice-simple-decoder backs gmm-latgen-simple
   gmm-latgen-simple --beam=16.0 --lattice-beam=10.0 --acoustic-scale=0.083 \
       final.mdl HCLG.fst "scp:feats.scp" "ark:|gzip -c > lat.gz"

Mapping
^^^^^^^

=======================================  ==========================================
kaldi                                    python API
=======================================  ==========================================
``LatticeSimpleDecoderConfig::beam``     ``kd.LatticeSimpleDecoderConfig(beam=...)``
``--beam``                               ``cfg.beam``
``--lattice-beam``                       ``cfg.lattice_beam``
``--prune-interval``                     ``cfg.prune_interval``
``--determinize-lattice``                ``cfg.determinize_lattice``
``--beam-ratio``                         ``cfg.beam_ratio``
``Decode(decodable)``                    ``decode(decodable)``
``GetBestPath(&lat, use_final)``         ``get_best_path(use_final_probs=True)``
``GetRawLattice(&lat, use_final)``       ``get_raw_lattice(use_final_probs=True)``
``ReachedFinal()``                       ``reached_final()``
``FinalRelativeCost()``                  ``final_relative_cost()``
=======================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.LatticeSimpleDecoderConfig
   :members:
   :special-members: __init__

.. autoclass:: kaldi_decoder.LatticeSimpleDecoder
   :members:
   :special-members: __init__
