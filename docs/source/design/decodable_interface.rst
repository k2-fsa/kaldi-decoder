The DecodableInterface boundary
===============================

kaldi splits every decoder from every acoustic model with one tiny abstract
class (:kaldi:`src/itf/decodable-itf.h`).  kaldi-decoder keeps that boundary
verbatim -- and it is why this library contains no GMM/NN code at all.

Four methods
------------

* ``LogLikelihood(frame, index)`` -- one scalar query.
* ``NumFramesReady()`` -- how far the future is visible (streaming).
* ``IsLastFrame(frame)`` -- end-of-stream signal.
* ``NumIndices()`` -- label space size.

That is the entire contract.  GMMs (``DecodableAmDiagGmmScaled`` in
kaldi-hmm-gmm), matrices (:doc:`../python_api/decodable_matrix`), CTC
outputs, and Python prototypes all fit.

Why per-(frame, index) queries
------------------------------

The decoder decides WHICH indices to ask for (only arcs leaving active
states), so a decodable never pays for unused scores.  Matrix-backed
decodables make each query O(1); models that score on demand (streaming NN)
can do exactly that.  The price is a virtual call per query -- acceptable
next to the decoder's own bookkeeping, and eliminated entirely in C++-only
builds (devirtualized when the concrete type is visible).

One-based indices
-----------------

Openfst labels are one-based (0 is epsilon), so transition-ids -- which are
graph input labels -- are one-based too.  Keeping the decodable's index
space identical to the label space avoids a mapping at every query.

TransitionInformation
---------------------

The one thing a decodable may additionally want is the transition model's
tid->pdf map (GMM scores are per pdf, decoders ask per tid).  kaldi passes
this as a separate abstract class,
:class:`~kaldi_decoder.TransitionInformation` -- optional, and only needed by
the mapped matrix decodables and the utterance wrappers.  This keeps the
decodable interface itself free of any model structure.

Ownership
---------

In C++ decoders borrow both the graph and the decodable (no hidden copies;
the caller owns them).  In the Python bindings the graph is tied to the
decoder with ``py::keep_alive`` -- a temporary graph stays alive until the
decoder is destroyed (see :kd:`test_lifetime.py
<kaldi-decoder/python/tests/test_lifetime.py>`); the decodable is only
needed during the decode calls themselves.
