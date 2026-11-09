FasterDecoder
=============

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/faster-decoder.h`; kaldi counterpart:
:kaldi:`src/decoder/faster-decoder.h <src/decoder/faster-decoder.h>`.

.. code-block:: cpp

   kaldi_decoder::FasterDecoderOptions opts;   // beam=16.0, ...
   kaldi_decoder::FasterDecoder decoder(graph, opts);
   decoder.InitDecoding();
   decoder.AdvanceDecoding(&decodable);
   fst::Lattice best;
   decoder.GetBestPath(&best);
