DecodableInterface
==================

The scoring interface every decoder consumes, matching kaldi's
:kaldi:`DecodableInterface <src/itf/decodable-itf.h>` (see
:kd:`kaldi-decoder/csrc/decodable-itf.h`).  Implement it in Python to plug any
acoustic model into the decoders; the pybind11 trampoline keeps virtual
dispatch on the Python side.

A decodable answers one question: *how likely is transition-id ``index`` at
frame ``frame``?*  Indices are **one-based** (openfst convention).

Usage
-----

.. code-block:: python

   import kaldi_decoder as kd

   class MyDecodable(kd.DecodableInterface):
       def num_frames_ready(self) -> int:
           return 100

       def is_last_frame(self, frame: int) -> bool:
           return frame == self.num_frames_ready() - 1

       def num_indices(self) -> int:
           return 30          # number of transition-ids (1-based)

       def log_likelihood(self, frame: int, index: int) -> float:
           # higher is better; return the acoustic log-likelihood of
           # transition-id `index` at `frame`.
           return my_scores[frame, index - 1]

   decodable = MyDecodable()
   decoder = kd.FasterDecoder(graph, kd.FasterDecoderOptions(beam=16.0))
   decoder.init_decoding()
   decoder.advance_decoding(decodable)

**Lifetime rule:** decoders hold a reference to the decodable (and to the
graph) without owning it.  The decoder keeps the graph alive automatically; the
decodable is only needed during the decode calls.

Equivalent in kaldi
-------------------

Any kaldi decoder binary takes scores from a GMM/NN model internally; the
DecodableInterface is exactly the seam kaldi uses between the acoustic model
and the decoder:

.. code-block:: bash

   # kaldi computes these scores internally via DecodableAmDiagGmmScaled etc.
   gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 \
       --acoustic-scale=0.083 final.mdl HCLG.fst \
       "scp:split.1/feats.scp" "ark:|gzip -c > lat.1.gz"

Mapping
^^^^^^^

================================  ==========================================
kaldi                             python API
================================  ==========================================
``DecodableInterface``            ``kaldi_decoder.DecodableInterface``
``LogLikelihood(frame, index)``   ``log_likelihood(self, frame, index)``
``NumFramesReady()``              ``num_frames_ready(self)``
``IsLastFrame(frame)``            ``is_last_frame(self, frame)``
``NumIndices()``                  ``num_indices(self)``
================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.DecodableInterface
   :members:
   :special-members: __init__
