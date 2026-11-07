LatticeSimpleDecoder
====================

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/lattice-simple-decoder.h`; kaldi counterpart:
:kaldi:`src/decoder/lattice-simple-decoder.h
<src/decoder/lattice-simple-decoder.h>`.

.. code-block:: cpp

   kaldi_decoder::LatticeSimpleDecoderConfig cfg;
   kaldi_decoder::LatticeSimpleDecoder decoder(graph, cfg);
   decoder.Decode(&decodable);
   fst::Lattice best, raw;
   decoder.GetBestPath(&best, true);
   decoder.GetRawLattice(&raw, true);
