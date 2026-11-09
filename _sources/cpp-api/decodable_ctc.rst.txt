DecodableCtc
============

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/decodable-ctc.h`; original class (no kaldi
counterpart).  Matrix-backed decodable for CTC-style per-frame log-probs.

.. code-block:: cpp

   kaldi_decoder::FloatMatrix log_probs(2, 2);
   log_probs << -0.1f, -5.0f, -5.0f, -0.1f;
   kaldi_decoder::DecodableCtc dec(log_probs);   // copies
   float s = dec.LogLikelihood(0, 1);            // -0.1
