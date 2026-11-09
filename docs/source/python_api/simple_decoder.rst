SimpleDecoder
=============

A Viterbi decoder over a static FST, matching kaldi's
:kaldi:`SimpleDecoder <src/decoder/simple-decoder.h>` (see
:kd:`kaldi-decoder/csrc/simple-decoder.h`).  It keeps one token per FST state
(no beam pruning of states beyond the beam value) and is best for small
graphs or as a reference decoder.

Usage
-----

.. code-block:: python

   import kaldifst
   import kaldi_decoder as kd

   graph = kaldifst.StdVectorFst.read("HCLG.fst")   # keep it alive!
   decodable = MyDecodable()                        # see DecodableInterface

   decoder = kd.SimpleDecoder(graph, beam=16.0)
   ok = decoder.decode(decodable)          # True if tokens survive to the end
   decoder.reached_final()                 # True if a final state was reached
   ok, best = decoder.get_best_path()      # best path as a kaldifst Lattice
   decoder.final_relative_cost()           # inf => no final state reached

**Lifetime:** the decoder holds a reference to ``graph`` (released with the decoder); temporaries are safe.

Equivalent in kaldi
-------------------

.. code-block:: bash

   # kaldi's simple-decoder is used by gmm-decode-simple etc.
   gmm-decode-simple --beam=16.0 final.mdl graph.fst \
       "scp:feats.scp" ark:tra.ark

Mapping
^^^^^^^

================================  ==========================================
kaldi                             python API
================================  ==========================================
``SimpleDecoder(fst, beam)``      ``kd.SimpleDecoder(graph, beam)``
``Decode(decodable)``             ``decode(decodable)``
``ReachedFinal()``                ``reached_final()``
``GetBestPath(&lat, use_final)``  ``get_best_path(use_final_probs=True)``
``FinalRelativeCost()``           ``final_relative_cost()``
================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.SimpleDecoder
   :members:
   :special-members: __init__
