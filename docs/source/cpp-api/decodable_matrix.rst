Decodable matrix classes
========================

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/decodable-matrix.h`; kaldi counterpart:
:kaldi:`src/decoder/decodable-matrix.h <src/decoder/decodable-matrix.h>`.
Precomputed score matrices as decodables (see
:doc:`../design/decodable_interface`).

.. code-block:: cpp

   kaldi_decoder::FloatMatrix likes(100, 11);  // frames x pdfs
   // ... fill likes ...
   kaldi_decoder::DecodableMatrixScaledMapped dec(tm, likes, 0.083f);
   float s = dec.LogLikelihood(0, 1);  // scale * likes(0, tm tid 1's pdf)
