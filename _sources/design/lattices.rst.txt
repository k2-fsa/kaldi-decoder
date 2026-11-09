Lattice weights
===============

kaldi lattices use ``LatticeWeight = (Value1, Value2)`` where **Value1 is the
graph cost** (LM + transition probabilities) and **Value2 is the acoustic
cost** -- both negated log-probabilities (lower = better).  Keeping them
separate is the whole point: rescoring or reweighting one side needs no
re-decoding.

Semiring
--------

``Plus`` (alternative paths) minimizes ``Value1 + Value2``; ties break by
smaller Value1 (see ``Compare`` in
:kaldi:`fstext/lattice-weight.h <src/fstext/lattice-weight.h>`).
``Times`` (concatenation) adds component-wise.  So a lattice is a
shortest-path problem over a pair of tropical costs.

Scaling
-------

``ScaleLattice(LatticeScale(lmwt, acwt), &lat)`` multiplies Value1 by
``lmwt`` and Value2 by ``acwt``.  kaldi conventions:

* ``gmm-latgen-faster`` writes lattices with acoustic costs divided by
  ``--acoustic-scale`` (``ScaleLattice(1, 1/acoustic_scale)``) so the
  acoustic side is in raw log-likelihood units;
* ``lattice-best-path --lm-scale=X --acoustic-scale=Y`` then finds the
  1-best of the rescaled lattice: ``argmin(X*V1 + Y*V2)``.

:func:`kaldi_decoder.lattice_best_path` is exactly that core.  The decode
wrappers return lattices already scaled the kaldi way, so calling
``lattice_best_path(lat, lm_scale, 1.0)`` reproduces kaldi's scoring.

Why not determinize here
------------------------

Determinization (``DeterminizeLatticePhonePruned``) merges equivalent
alternative paths and is what kaldi writes to ``compact lattices``.  It does
NOT change the 1-best, so this port leaves it out (the raw lattice is enough
for best-path / n-best extraction); the option structs are still provided for
callers that want to plug in determinization from
`kaldifst <https://github.com/k2-fsa/kaldifst>`_.

Alignment extraction
--------------------

A lattice path's input labels are transition-ids (the alignment) and its
output labels are words.  ``GetLinearSymbolSequence`` linearizes a path FST
into exactly those two sequences plus the total weight -- the single place
all our outputs (words/alignments/likes) come from.
