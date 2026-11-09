Decoding an utterance
=====================

This walks through decoding one utterance end to end: build (or load) a
decoding graph, wrap your acoustic scores, run
:class:`~kaldi_decoder.LatticeFasterDecoder`, and extract results.

1. The decoding graph
---------------------

The graph ``HCLG`` maps transition-ids (decoder input labels) to words
(output labels).  kaldi-decoder never builds graphs -- use
`kaldifst <https://github.com/k2-fsa/kaldifst>`_ or kaldi's graph-building
tools:

.. code-block:: python

   import kaldifst

   graph = kaldifst.StdVectorFst.read("HCLG.fst")
   print(graph.num_states)

You can also construct tiny graphs by hand (as
:kd:`the C++ unit tests <kaldi-decoder/csrc/lattice-faster-decoder-test.cc>`
and :kd:`the python tests <kaldi-decoder/python/tests/test_decoders.py>` do):

.. code-block:: python

   g = kaldifst.StdVectorFst()
   s0, s1, s2 = g.add_state(), g.add_state(), g.add_state()
   g.start = s0
   g.add_arc(s0, kaldifst.StdArc(1, 10, 0.0, s1))   # tid 1 -> word 10
   g.add_arc(s1, kaldifst.StdArc(2, 20, 0.0, s2))   # tid 2 -> word 20
   g.set_final(s2, 0.0)

.. note::
   The decoder **holds a reference** to the graph (``py::keep_alive``), so
   the graph is released only when the decoder is destroyed -- passing a
   temporary graph is safe.  (The same rule is enforced by the regression
   test :kd:`test_lifetime.py
   <kaldi-decoder/python/tests/test_lifetime.py>`.)

2. The scores (a decodable)
---------------------------

Pick whichever fits your scores; see :doc:`../usage/decodables` for the full
guide.  The quickest option when you already have a score matrix:

.. code-block:: python

   import numpy as np

   likes = np.random.randn(num_frames, num_pdfs).astype(np.float32)
   decodable = kd.DecodableMatrixScaledMapped(tm, likes, scale=0.083)

3. Decode
---------

.. code-block:: python

   cfg = kd.LatticeFasterDecoderConfig(beam=13.0, lattice_beam=6.0,
                                       max_active=7000)
   decoder = kd.LatticeFasterDecoder(graph, cfg)
   decoder.decode(decodable)

   if not decoder.reached_final():
       print("warning: no final state reached; consider allow_partial")

4. Get results
--------------

.. code-block:: python

   ok, best = decoder.get_best_path()
   # best is a kaldifst.Lattice; read the 1-best labels by walking
   # start -> final (a ready-made walker lives in the unit tests)

   ok, raw = decoder.get_raw_lattice()   # alternatives within lattice_beam

Or let the utterance-level wrapper do steps 3-4 and return plain lists (this
is what kaldi's ``gmm-latgen-faster`` writes to its tables):

.. code-block:: python

   ok, alignment, words, like, lat = kd.decode_utterance_lattice_faster(
       decoder=decoder, decodable=decodable, trans_model=tm,
       utt="utt1", allow_partial=False, acoustic_scale=0.083)

5. Score the lattice (lm weights / word penalties)
--------------------------------------------------

``lat`` is returned exactly as kaldi writes ``lat.gz`` (acoustic costs
divided by ``acoustic_scale``).  To sweep lm weights and word insertion
penalties like kaldi's ``local/score.sh``:

.. code-block:: python

   best = None
   for lmwt in (0.5, 0.75, 1.0, 1.25):
       ok, ali, words, like = kd.lattice_best_path(
           lat, lm_scale=lmwt, acoustic_scale=1.0)
       if best is None or like > best[0]:
           best = (like, lmwt, words)
   print("best lmwt:", best[1], "words:", best[2])

Choosing beam settings
----------------------

* ``beam`` -- width of the active-token beam (more = slower, more accurate).
* ``lattice_beam`` -- how much worse than the best path an alternative may
  be to stay in the output lattice.
* ``max_active`` / ``min_active`` -- hard cap / floor on active tokens per
  frame (the decoder adapts the effective beam between them).

kaldi's typical settings for GMM decoding are ``beam=13.0``,
``lattice_beam=6.0``, ``max_active=7000``; for alignment
``beam=200`` (see :doc:`alignment`).
