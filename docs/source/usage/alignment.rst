Forced alignment
================

Forced alignment finds the best state sequence for a KNOWN word sequence --
the engine of kaldi's ``gmm-align-compiled``.  The decoding graph is the
per-utterance training graph (transition-ids on the input side, the known
words on the output side), compiled from the lexicon + transcript.

1. Build the per-utterance graph
--------------------------------

Use kaldi's ``compile-train-graphs`` (or
`kaldi-hmm-gmm <https://github.com/csukuangfj/kaldi-hmm-gmm>`_
``TrainingGraphCompiler``).  IMPORTANT -- kaldi compiles these graphs with
``transition_scale=0.0 self_loop_scale=0.0``: the transition probabilities
are added ONCE at alignment time (they change every training iteration).

2. Align
---------

.. code-block:: python

   import kaldifst
   import kaldi_decoder as kd

   graph = kaldifst.StdVectorFst.read("train_graph.1.fst")  # keep alive!
   decodable = kd.DecodableMatrixScaledMapped(tm, likes, scale=0.1)

   cfg = kd.AlignConfig(beam=200.0, retry_beam=0.0, careful=False)
   (num_done, num_error, num_retried, tot_like, frame_count,
    alignment, words) = kd.align_utterance_wrapper(
        config=cfg, utt="utt1", acoustic_scale=0.1,
        fst=graph, decodable=decodable,
        num_done=0, num_error=0, num_retried=0,
        tot_like=0.0, frame_count=0)

   print("alignment (transition-ids):", alignment)
   print("frames per state:", frame_count)

The five counters are inputs that the wrapper *increments* (they mirror
kaldi's accumulating table writers), so a loop over utterances just threads
them through.

Beam settings
-------------

* ``beam=200`` -- alignment uses a wide beam: the answer is known, we just
  need to find it.
* ``retry_beam`` -- if the first attempt fails (no path), retry with this
  beam; 0 disables the retry (kaldi's ``--retry-beam``).
* ``careful=True`` -- kaldi's ``--careful``: appends the graph to itself so a
  too-fast traversal ("blind alley") is detected instead of silently
  producing a wrong alignment.  Slower; use when failures matter more than
  speed.  NOTE: careful mode MODIFIES the graph (adds the loop) -- pass a
  copy if you want to reuse it.

Transition probabilities
------------------------

The graph must be compiled WITHOUT transition probabilities and they are
added inside the alignment call (exactly kaldi's
``gmm-align-compiled``):  with the scales given to the *compile* step at
zero.  If you bake them into the graph AND add them here you get them
double-counted (a real bug we hit; see the kaldi-hmm-gmm plan notes).

Where the alignment comes from
------------------------------

Internally: decode the graph (Viterbi), take the decode-time best path, and
read its input labels (transition-ids) and output labels (words) -- see
:doc:`../design/decoders`.
