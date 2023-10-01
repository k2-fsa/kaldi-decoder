# Introduction

This repository ports the C++ decoder code from [Kaldi][Kaldi]
using [OpenFst][OpenFst] without depending on [Kaldi][Kaldi].

# Usage

Please first install it using

```bash
pip install kaldi-decoder
```

and see the following examples about CTC decoding:

|Description| URL|
|---|---|
|Decoding with H|https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/jit_pretrained_decode_with_H.py|
|Decoding with HL|https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/jit_pretrained_decode_with_HL.py|
|Decoding with HLG|https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/jit_pretrained_decode_with_HLG.py|

[Kaldi]: https://github.com/kaldi-asr/kaldi
[OpenFst]: https://www.openfst.org/

