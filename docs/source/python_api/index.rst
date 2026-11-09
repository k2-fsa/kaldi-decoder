Python API
==========

The Python bindings live in the :mod:`kaldi_decoder` module (the extension is
``kaldi_decoder/lib/_kaldi_decoder``).  Building graphs needs the
`kaldifst <https://github.com/k2-fsa/kaldifst>`_ package; kaldi-decoder itself
does not construct FSTs.

Each page below has a usage example, the equivalent kaldi commands, a
kaldi-flag mapping table, and the full API reference.

.. toctree::
   :maxdepth: 2

   decodable_itf
   decodable_ctc
   decodable_matrix
   transition_information
   simple_decoder
   faster_decoder
   lattice_simple_decoder
   lattice_faster_decoder
   decoder_wrappers
   reference
