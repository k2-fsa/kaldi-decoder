C++ API
=======

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

The headers live in :kd:`kaldi-decoder/csrc/`.  Link the static library
``kaldi-decoder-core``; it depends only on kaldifst/openfst and Eigen.  The
pages below show usage examples per module; the full API reference is the
doxygen HTML linked above.

.. toctree::
   :maxdepth: 1

   decodable_itf
   decodable_ctc
   decodable_matrix
   transition_information
   simple_decoder
   faster_decoder
   lattice_simple_decoder
   lattice_faster_decoder
   decoder_wrappers
