DecodableCtc
============

A matrix-backed decodable for CTC-style scores, matching the spirit of
kaldi's :kaldi:`DecodableMatrixScaled <src/decoder/decodable-matrix.h>` but
without any transition-model mapping (see
:kd:`kaldi-decoder/csrc/decodable-ctc.h`).  You hand in a ``(num_frames,
num_classes)`` matrix of log-probabilities; the class copies it.

**Indices are one-based**: ``log_likelihood(t, 1)`` is column 0.  This is for
compatibility with openfst labels.

Usage
-----

.. code-block:: python

   import numpy as np
   import kaldi_decoder as kd

   # 2 frames x 2 classes of log-probs
   log_probs = np.array([[-0.1, -5.0],
                         [-5.0, -0.1]], dtype=np.float32)

   decodable = kd.DecodableCtc(log_probs, offset=0)
   decodable.num_frames_ready()          # 2
   decodable.num_indices()               # 2
   decodable.log_likelihood(0, 1)        # -0.1  (frame 0, class 0)
   decodable.log_likelihood(1, 2)        # -0.1  (frame 1, class 1)

   # `offset` lets row 0 of the matrix correspond to frame `offset` of the
   # utterance (chunked streaming).
   decodable = kd.DecodableCtc(log_probs, offset=3)   # frames 3..4

Equivalent in kaldi
-------------------

kaldi has no CTC decodable; the closest is feeding precomputed per-frame
scores to a decodable-matrix class:

.. code-block:: bash

   # kaldi equivalent: per-frame scores in a matrix, consumed by
   # DecodableMatrixScaled (src/decoder/decodable-matrix.h)

Mapping
^^^^^^^

================================  ==========================================
kaldi                             python API
================================  ==========================================
``Matrix<BaseFloat> likes``       ``DecodableCtc(log_probs, offset=0)``
(matrix row = frame)              (``numpy.ndarray``, float32)
frame offset (chunked input)      ``DecodableCtc(..., offset=n)``
================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.DecodableCtc
   :members:
   :special-members: __init__
