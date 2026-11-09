How the decoders work
=====================

All four decoders are **token-passing** algorithms on the decoding FST: at
each frame, every active state holds a *token* (the best cost so far plus a
backpointer); emitting arcs extend tokens to the next frame, epsilon arcs
are consumed within the frame, and a beam prunes hopeless states.

Why token passing
-----------------

The decoding graph ``HCLG`` is a transducer whose input labels are
transition-ids and whose arcs already carry graph costs (LM + transition
probabilities).  Acoustic scores arrive per frame, so the total cost of a
path is ``graph_cost + acoustic_cost`` accumulated left to right -- exactly
the tropical semiring shortest-path problem, solved online by dynamic
programming (Viterbi).  Token passing keeps only the best token per state,
which is sufficient for the 1-best and, with extra bookkeeping, for lattices.

Beam pruning
------------

``beam`` is subtracted from the best cost of the frame to form the pruning
threshold: any token worse than ``best + beam`` dies.  ``max_active`` /
``min_active`` bound the number of surviving tokens (the decoder adapts the
effective beam to hit the bounds).  This makes cost -- not graph size -- the
dominant factor.

Emitting vs non-emitting steps
------------------------------

One "frame" of decoding is: ProcessEmitting (consume one frame of acoustic
scores via transition-id arcs) followed by ProcessNonemitting (closure over
epsilon arcs).  The final costs of a frame's tokens are only complete after
the non-emitting closure, which is why ``GetBestPath`` after the last frame
may consult final-probs (``use_final_probs``).

Self-loops
----------

HMM self-loops are ordinary arcs in the graph (inserted by kaldi's
``add-self-loops``).  ``FasterDecoder`` assumes the graph is self-loop-free
and is fastest; ``LatticeFasterDecoder`` / ``LatticeSimpleDecoder`` handle
general graphs.

Lattice decoders: forward links
-------------------------------

To output a lattice (alternatives), a token must remember more than its best
predecessor: every near-best predecessor within ``lattice_beam`` becomes a
*forward link*.  ``LatticeFasterDecoder`` maintains these links per token
and prunes them lazily (``PruneForwardLinks``/``PruneTokensForFrame``), with
final links computed at the end (``FinalizeDecoding``) so that final-probs
are accounted exactly once.  Cost offsets (``cost_offsets_``) renormalize
per-frame costs to keep floats in range during long utterances.

Why two beams
-------------

``beam`` controls how many states stay alive (speed/accuracy of the 1-best);
``lattice_beam`` controls how much detail the output lattice keeps.  Keeping
them separate lets you decode fast but still dump a rich lattice (or vice
versa).

GetBestPath semantics
---------------------

``GetBestPath`` returns the DECODE-TIME best path (the Viterbi winner under
the costs the decoder saw).  This is what kaldi writes to its words and
alignment tables in ``gmm-latgen-faster``.  Scoring a lattice under
different lm weights is a separate step
(:doc:`lattices` / ``lattice_best_path``).

These designs all come straight from kaldi; see
:kaldi:`src/decoder/lattice-faster-decoder.cc
<src/decoder/lattice-faster-decoder.cc>` and kaldi's ``doc/hmm.dox`` for the
original descriptions.
