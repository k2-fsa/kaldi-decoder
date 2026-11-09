Decoder wrappers
================

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/decoder-wrappers.h`; kaldi counterpart:
:kaldi:`src/decoder/decoder-wrappers.h
<src/decoder/decoder-wrappers.h>` (the binary-level parts of
``gmm-align-compiled`` / ``gmm-latgen-faster``) and
:kaldi:`src/latbin/lattice-best-path.cc <src/latbin/lattice-best-path.cc>`.

.. code-block:: cpp

   kaldi_decoder::AlignConfig cfg;   // beam=200, retry_beam=0, careful=false
   std::vector<int32_t> alignment, words;
   kaldi_decoder::AlignUtteranceWrapper(cfg, "utt", 0.1f, &graph, &decodable,
                                        &num_done, &num_error, &num_retried,
                                        &tot_like, &frame_count,
                                        &alignment, &words);
