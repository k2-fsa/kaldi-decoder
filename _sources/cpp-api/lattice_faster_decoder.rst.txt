LatticeFasterDecoder
====================

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/lattice-faster-decoder.h`; kaldi counterpart:
:kaldi:`src/decoder/lattice-faster-decoder.h
<src/decoder/lattice-faster-decoder.h>`.  The decoder behind
``gmm-latgen-faster``; design notes in :doc:`../design/decoders`.

.. code-block:: cpp

   kaldi_decoder::LatticeFasterDecoderConfig cfg;
   cfg.beam = 13.0;
   cfg.lattice_beam = 6.0;
   kaldi_decoder::LatticeFasterDecoder decoder(graph, cfg);
   decoder.Decode(&decodable);
   fst::Lattice best, raw;
   decoder.GetBestPath(&best, true);
   decoder.GetRawLattice(&raw, true);
