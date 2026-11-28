LatticeFasterOnlineDecoder
==========================

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/lattice-faster-online-decoder.h`; kaldi
counterpart: :kaldi:`src/decoder/lattice-faster-online-decoder.h
<src/decoder/lattice-faster-online-decoder.h>`.

.. code-block:: cpp

   kaldi_decoder::LatticeFasterOnlineDecoder decoder(graph, cfg);
   decoder.InitDecoding();
   decoder.AdvanceDecoding(&decodable);
   // one transition-id per decoded frame (not in kaldi; the loop of
   // OnlineSilenceWeighting::ComputeCurrentTraceback)
   std::vector<int32_t> tids =
       kaldi_decoder::GetBestPathTransitionIds(decoder, false);
   // kaldi's own iterator API
   auto iter = decoder.BestPathEnd(false);
   fst::LatticeArc arc;
   while (!iter.Done()) iter = decoder.TraceBackBestPath(iter, &arc);
