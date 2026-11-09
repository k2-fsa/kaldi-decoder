Writing a decodable
===================

Everything a decoder knows about acoustics comes through
:class:`~kaldi_decoder.DecodableInterface`.  Three ways to provide scores,
fastest first:

1. Matrix-backed (preferred when scores are computed in one shot)
-----------------------------------------------------------------

.. code-block:: python

   likes = ...  # (num_frames, num_pdfs) float32 numpy array
   decodable = kd.DecodableMatrixScaledMapped(tm, likes, scale=0.083)

All ``log_likelihood`` calls stay in C++.  The matrix is copied once at
construction (owned by the decodable); you may free the numpy array.

2. Python subclass (any custom scoring)
---------------------------------------

.. code-block:: python

   class MyDecodable(kd.DecodableInterface):
       def __init__(self, model, feats):
           super().__init__()
           self.scores = model.score(feats)   # compute any way you like

       def num_frames_ready(self):
           return self.scores.shape[0]

       def is_last_frame(self, frame):
           return frame == self.num_frames_ready() - 1

       def num_indices(self):
           return self.scores.shape[1]        # 1-based transition-ids

       def log_likelihood(self, frame, index):
           return self.scores[frame, index - 1]

Every ``log_likelihood`` call crosses the C++ -> Python boundary, so prefer
option 1 for large jobs; use this for prototyping or exotic scoring.

3. C++ at compile time
----------------------

For production systems, implement ``DecodableInterface`` in C++ inside your
own project and link ``kaldi-decoder-core`` (see the C++ API pages).  No
Python overhead at all.

Contract
--------

* Indices are **one-based** (openfst label convention); ``num_indices()``
  is the number of valid indices.
* ``log_likelihood(frame, index)`` returns the **acoustic log-likelihood**
  (higher = better); the decoder adds graph costs separately.
* ``NumFramesReady`` may grow over time (streaming); the decoder will not
  query beyond it.
* Keep the decodable alive for the whole decode (decoders store a pointer).

Design note: the interface is intentionally tiny (4 methods) so that any
acoustic model -- GMM, NN, CTC, cross-lingual hybrids -- can drive the same
decoders.  See :doc:`../design/decodable_interface`.
