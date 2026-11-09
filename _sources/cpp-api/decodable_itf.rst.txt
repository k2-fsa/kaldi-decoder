DecodableInterface
==================

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :kd:`kaldi-decoder/csrc/decodable-itf.h`; kaldi counterpart:
:kaldi:`src/itf/decodable-itf.h <src/itf/decodable-itf.h>`.  The scoring
interface every decoder consumes (see :doc:`../design/decodable_interface`).

.. code-block:: cpp

   class MyDecodable : public kaldi_decoder::DecodableInterface {
    public:
     float LogLikelihood(int32_t frame, int32_t index) override {
       return scores_(frame, index - 1);  // indices are one-based
     }
     int32_t NumFramesReady() const override { return scores_.rows(); }
     bool IsLastFrame(int32_t frame) const override {
       return frame == NumFramesReady() - 1;
     }
     int32_t NumIndices() const override { return scores_.cols(); }
   };
