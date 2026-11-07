FasterDecoder
=============

kaldi's :kaldi:`FasterDecoder <src/decoder/faster-decoder.h>` (see
:kd:`kaldi-decoder/csrc/faster-decoder.h`): a beam-pruned Viterbi decoder for
FSTs without self-loops (e.g. transducers built with
``add-self-loops``-style graphs need a lattice decoder instead).  It is the
fastest of the three simple decoders.

Usage
-----

.. code-block:: python

   import kaldifst
   import kaldi_decoder as kd

   graph = kaldifst.StdVectorFst.read("HCLG.fst")   # keep it alive!
   decodable = MyDecodable()

   opts = kd.FasterDecoderOptions(beam=16.0, max_active=100, min_active=20,
                                  beam_delta=0.5, hash_ratio=2.0)
   decoder = kd.FasterDecoder(graph, opts)
   decoder.init_decoding()
   decoder.advance_decoding(decodable)      # or advance one frame at a time:
   decoder.advance_decoding(decodable, max_num_frames=1)
   decoder.num_frames_decoded()
   decoder.reached_final()
   ok, best = decoder.get_best_path()

   # change settings mid-stream
   decoder.set_options(kd.FasterDecoderOptions(beam=8.0))

Equivalent in kaldi
-------------------

.. code-block:: bash

   # kaldi's faster-decoder backs e.g. gmm-decode-faster
   gmm-decode-faster --beam=16.0 --max-active=100 final.mdl graph.fst \
       "scp:feats.scp" ark:tra.ark

Mapping
^^^^^^^

===================================  ==========================================
kaldi                                python API
===================================  ==========================================
``FasterDecoderOptions::beam``       ``kd.FasterDecoderOptions(beam=...)``
``--beam``                           ``opts.beam``
``--max-active``                     ``opts.max_active``
``--min-active``                     ``opts.min_active``
``--beam-delta``                     ``opts.beam_delta``
``--hash-ratio``                     ``opts.hash_ratio``
``InitDecoding()``                   ``init_decoding()``
``AdvanceDecoding(d, n)``            ``advance_decoding(d, max_num_frames=-1)``
``GetBestPath(&lat)``                ``get_best_path(use_final_probs=True)``
===================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.FasterDecoderOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_decoder.FasterDecoder
   :members:
   :special-members: __init__
