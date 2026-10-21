Installation
============

We provide pre-built wheels for the following platforms:

.. list-table::
   :header-rows: 1

   * - Platform
     - Architecture
   * - Linux
     - x64, arm64
   * - macOS
     - x64, arm64, universal2
   * - Windows
     - x86 (32-bit), x64, arm64
   * - Android (Termux)
     - arm (armeabi-v7a), aarch64 (arm64-v8a), i686 (x86), x86_64

CPython 3.8 to 3.15 are supported.

Install from PyPI
-----------------

The easiest way is to install it from PyPI:

.. code-block:: bash

   pip install kaldi-decoder

or, if you prefer `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv pip install kaldi-decoder

Build from source
-----------------

Please first install ``cmake``, a C++ compiler and ``python3``.

.. code-block:: bash

   git clone https://github.com/k2-fsa/kaldi-decoder
   cd kaldi-decoder
   python3 setup.py install

If you want to build only the C++ libraries, without the Python bindings:

.. code-block:: bash

   git clone https://github.com/k2-fsa/kaldi-decoder
   cd kaldi-decoder
   mkdir build
   cd build
   cmake -DKALDI_DECODER_BUILD_PYTHON=OFF ..
   make -j4

To also run the C++ tests:

.. code-block:: bash

   cmake -DKALDI_DECODER_ENABLE_TESTS=ON ..
   make -j4
   ctest --output-on-failure

Usage
-----

Please see the following examples about CTC decoding:

.. list-table::
   :header-rows: 1

   * - Description
     - URL
   * - Decoding with H
     - `jit_pretrained_decode_with_H.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/jit_pretrained_decode_with_H.py>`_
   * - Decoding with HL
     - `jit_pretrained_decode_with_HL.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/jit_pretrained_decode_with_HL.py>`_
   * - Decoding with HLG
     - `jit_pretrained_decode_with_HLG.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/jit_pretrained_decode_with_HLG.py>`_
