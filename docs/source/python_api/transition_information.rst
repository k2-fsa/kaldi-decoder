TransitionInformation
=====================

The transition-model interface the decode wrappers use for tid->pdf mapping
and phone queries, matching kaldi's
:kaldi:`TransitionInformation <src/itf/transition-information.h>` (see
:kd:`kaldi-decoder/csrc/transition-information.h`).  Implement it in Python to
use :doc:`decoder_wrappers` with any transition model, or use
``kaldi_hmm_gmm.TransitionModel`` from the
`kaldi-hmm-gmm <https://github.com/csukuangfj/kaldi-hmm-gmm>`_ package, which
implements the same interface in C++.

Usage
-----

.. code-block:: python

   import kaldi_decoder as kd

   class MyTm(kd.TransitionInformation):
       def transition_ids_equivalent(self, a: int, b: int) -> bool:
           return self.transition_id_to_pdf_array()[a] == \
                  self.transition_id_to_pdf_array()[b]

       def transition_id_is_start_of_phone(self, tid: int) -> bool:
           return tid in self.phone_starts

       def transition_id_to_phone(self, tid: int) -> int:
           return self.tid2phone[tid]

       def is_final(self, tid: int) -> bool:
           return tid in self.final_tids

       def is_self_loop(self, tid: int) -> bool:
           return tid in self.self_loop_tids

       def transition_id_to_pdf_array(self) -> list:
           # index 0 is unused; transition-ids are one-based
           return [-1, 0, 1, 1, 2, 3]

       def num_pdfs(self) -> int:
           return 4

**Note:** ``transition_id_to_pdf_array`` must return the *whole* map every
call (it is cached by the trampoline; the cached reference is what
decodables keep).

Equivalent in kaldi
-------------------

kaldi implements this interface in ``TransitionModel``
(:kaldi:`src/hmm/transition-model.h <src/hmm/transition-model.h>`), which is
built from a tree + topology:

.. code-block:: bash

   # kaldi: TransitionModel is read from a .mdl file
   gmm-info exp/mono0a/final.mdl

Mapping
^^^^^^^

======================================  ==========================================
kaldi                                   python API
======================================  ==========================================
``TransitionInformation``               ``kaldi_decoder.TransitionInformation``
``TransitionIdsEquivalent(a, b)``       ``transition_ids_equivalent(a, b)``
``TransitionIdIsStartOfPhone(tid)``     ``transition_id_is_start_of_phone(tid)``
``TransitionIdToPhone(tid)``            ``transition_id_to_phone(tid)``
``IsFinal(tid)``                        ``is_final(tid)``
``IsSelfLoop(tid)``                     ``is_self_loop(tid)``
``TransitionIdToPdfArray()``            ``transition_id_to_pdf_array()``
``NumPdfs()``                           ``num_pdfs()``
``NumTransitionIds()``                  ``num_transition_ids()``
======================================  ==========================================

API reference
-------------

.. autoclass:: kaldi_decoder.TransitionInformation
   :members:
   :special-members: __init__
