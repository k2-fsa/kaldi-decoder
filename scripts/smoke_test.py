#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation (author: Fangjun Kuang)
#
# Smoke test for the built wheels. Run by build-wheels-win-arm64.yaml and
# build-wheels-android.yaml after installing a freshly built wheel, and by
# test-pip-install.yaml / test-uv-install.yaml after installing from PyPI.
#
# It runs a real decode through FasterDecoder so that a wheel which cannot
# load its extension module (missing shared library, wrong architecture, wrong
# platform tag) fails here rather than on a user's machine.
#
# Building the decoding graph needs kaldifst (kaldi-decoder does not expose
# FST construction itself), so install it alongside kaldi-decoder:
#
#     pip install kaldi-decoder kaldifst

import platform
import sys

import kaldifst
import kaldi_decoder
from kaldi_decoder import DecodableInterface, FasterDecoder, FasterDecoderOptions

NUM_LABELS = 2
NUM_FRAMES = 2


def build_h() -> kaldifst.StdVectorFst:
    # A linear graph H: s0 --1/1--> s1 --2/2--> s2 (final)
    fst = kaldifst.StdVectorFst()
    s0 = fst.add_state()
    s1 = fst.add_state()
    s2 = fst.add_state()
    fst.start = s0

    fst.add_arc(s0, kaldifst.StdArc(ilabel=1, olabel=1, weight=0.0, nextstate=s1))
    fst.add_arc(s1, kaldifst.StdArc(ilabel=2, olabel=2, weight=0.0, nextstate=s2))
    fst.set_final(s2, weight=0.0)
    return fst


class FakeDecodable(DecodableInterface):
    """Hands out a deterministic score so the decoded path is checkable."""

    def num_frames_ready(self) -> int:
        return NUM_FRAMES

    def is_last_frame(self, frame: int) -> bool:
        return frame == NUM_FRAMES - 1

    def num_indices(self) -> int:
        return NUM_LABELS

    def log_likelihood(self, frame: int, index: int) -> float:
        # The correct label at frame t is t+1 (i.e. 1 then 2).
        return 0.0 if index == frame + 1 else -10.0


def main() -> int:
    print(
        "kaldi_decoder:",
        getattr(kaldi_decoder, "__version__", "<none>"),
        kaldi_decoder.__file__,
    )
    print("kaldifst:", getattr(kaldifst, "__version__", "<none>"), kaldifst.__file__)
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("machine:", platform.machine())

    # Exercise the plain bindings too -- these need no FST at all.
    options = FasterDecoderOptions(
        beam=16.0, max_active=100, min_active=20, beam_delta=0.5, hash_ratio=2.0
    )
    print("options:", str(options))
    assert "beam=16" in str(options), f"unexpected ToString(): {options}"
    options.beam = 8.0
    assert "beam=8" in str(options), f"beam setter did not take: {options}"

    # Both of these must stay alive in locals for as long as `decoder` is in
    # use. FasterDecoder keeps a reference to the FST without owning it, and
    # DecodableInterface is a pybind11 trampoline -- handing either one in as a
    # temporary lets it be collected while C++ still points at it, which
    # segfaults inside init_decoding()/advance_decoding().
    graph = build_h()
    decodable = FakeDecodable()

    decoder = FasterDecoder(graph, FasterDecoderOptions(beam=16.0))
    decoder.init_decoding()
    decoder.advance_decoding(decodable)

    print("num_frames_decoded:", decoder.num_frames_decoded())
    assert decoder.num_frames_decoded() == NUM_FRAMES, (
        f"expected {NUM_FRAMES} frames, got {decoder.num_frames_decoded()}"
    )

    assert decoder.reached_final(), "decoder did not reach a final state"

    ok, best_path = decoder.get_best_path()
    assert ok, "get_best_path() failed"
    print("best_path type:", type(best_path).__name__)

    text = best_path.to_str() if hasattr(best_path, "to_str") else str(best_path)
    print("best_path:")
    print(text)

    # OpenFst text format: "src ilabel olabel [weight] dest". The winning path
    # must consume labels 1 then 2, in that order, with ilabel == olabel.
    labels = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue  # final-state line, e.g. just "2"
        src, ilabel, olabel = fields[0], fields[1], fields[2]
        labels.append((ilabel, olabel))

    assert labels == [("1", "1"), ("2", "2")], f"unexpected best path: {labels}\n{text}"

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
