Decodable matrix classes
========================

Precomputed log-likelihood matrices wrapped as decodables, matching kaldi's
:kaldi:`decoder/decodable-matrix.h <src/decoder/decodable-matrix.h>` (see
:kd:`kaldi-decoder/csrc/decodable-matrix.h`).  Use these whenever the full
score matrix is already available -- they are the efficient alternative to a
per-call Python decodable: all ``log_likelihood`` calls stay in C++.

Four classes, same names as kaldi:

* ``DecodableMatrixScaledMapped`` -- acoustic scale + tid->pdf mapping
  (GMM/NN scores are per-**pdf**; decoders ask per transition-id).
* ``DecodableMatrixMapped`` -- same mapping, no scale, with a frame offset.
* ``DecodableMatrixMappedOffset`` -- incrementally fed chunks
  (multi-threaded decoding).
* ``DecodableMatrixScaled`` -- no mapping; columns are the indices directly.

Usage
-----

.. code-block:: python

   import numpy as np
   import kaldi_decoder as kd

   # tm: a kd.TransitionInformation (e.g. wrapping your transition model)
   likes = np.random.randn(100, 11).astype(np.float32)  # 100 frames x 11 pdfs

   decodable = kd.DecodableMatrixScaledMapped(tm, likes, scale=0.083)
   decodable.num_frames_ready()       # 100
   decodable.num_indices()            # tm.num_transition_ids()
   decodable.log_likelihood(0, 1)     # scale * likes[0, tid_to_pdf[1]]

   # chunked input (frames arrive in pieces):
   dec = kd.DecodableMatrixMappedOffset(tm)
   dec.accept_loglikes(chunk1, frames_to_discard=0)
   dec.accept_loglikes(chunk2, frames_to_discard=1)  # drop 1 old frame
   dec.input_is_finished()
   dec.first_available_frame()        # frames below this are gone

**Lifetime:** the constructors copy the numpy matrix and own the copy.  The
numpy array may be deleted immediately afterwards (see the regression test
:kd:`test_lifetime_after_numpy_gone
<kaldi-decoder/python/tests/test_decodable_matrix.py>`).

Equivalent in kaldi
-------------------

.. code-block:: bash

   # kaldi: the same classes consume a Matrix<BaseFloat> of per-pdf likes,
   # e.g. produced by a neural net and written with matrix-write.
   nnet3-compute ... | matrix-copy ark:- ark:likes.ark

Mapping
^^^^^^^

=================================  ===============================================
kaldi                              python API                                     
=================================  ===============================================
``Matrix<BaseFloat> likes``        ``numpy.ndarray`` (float32, copied)            
``DecodableMatrixScaledMapped``    ``kd.DecodableMatrixScaledMapped(tm, ...)``    
``DecodableMatrixMapped``          ``kd.DecodableMatrixMapped(tm, ...)``          
``AcceptLoglikes(m, d)``           ``accept_loglikes(chunk, frames_to_discard)``  
``InputIsFinished()``              ``input_is_finished()``                        
``FirstAvailableFrame()``          ``first_available_frame()``                    
``DecodableMatrixScaled``          ``kd.DecodableMatrixScaled(likes, scale)``     
=================================  ===============================================

API reference
-------------

.. autoclass:: kaldi_decoder.DecodableMatrixScaledMapped
   :members:
   :special-members: __init__

.. autoclass:: kaldi_decoder.DecodableMatrixMapped
   :members:
   :special-members: __init__

.. autoclass:: kaldi_decoder.DecodableMatrixMappedOffset
   :members:
   :special-members: __init__

.. autoclass:: kaldi_decoder.DecodableMatrixScaled
   :members:
   :special-members: __init__
