SimpleDecoder
=============

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/simple-decoder.h`; kaldi counterpart:
:kaldi:`src/decoder/simple-decoder.h <src/decoder/simple-decoder.h>`.

.. code-block:: cpp

   kaldi_decoder::SimpleDecoder decoder(graph, 16.0);
   decoder.Decode(&decodable);
   fst::Lattice best;
   if (decoder.GetBestPath(&best, true)) { /* ... */ }
