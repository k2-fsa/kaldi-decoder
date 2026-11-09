#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_lifetime_py
"""Regression tests: decoders hold a reference to the graph.

The py::keep_alive<1, 2>() call policy on every decoder constructor ties the
graph's lifetime to the decoder, so a temporary graph (or one explicitly
deleted) must NOT cause a use-after-free.
"""
import gc
import unittest

try:
    import kaldifst
except ImportError:
    raise SystemExit(77)  # optional dependency missing -> ctest "skipped"

from kaldi_decoder import (
    DecodableInterface,
    FasterDecoder,
    FasterDecoderOptions,
    LatticeFasterDecoder,
    LatticeFasterDecoderConfig,
    LatticeSimpleDecoder,
    LatticeSimpleDecoderConfig,
    SimpleDecoder,
)


class TableDecodable(DecodableInterface):
    def __init__(self):
        super().__init__()

    def num_frames_ready(self):
        return 2

    def is_last_frame(self, frame):
        return frame == 1

    def num_indices(self):
        return 2

    def log_likelihood(self, frame, index):
        return 0.0 if index == frame + 1 else -10.0


def build_graph():
    """A linear graph returned as a temporary (no caller reference)."""
    fst = kaldifst.StdVectorFst()
    s0 = fst.add_state()
    s1 = fst.add_state()
    s2 = fst.add_state()
    fst.start = s0
    fst.add_arc(s0, kaldifst.StdArc(1, 1, 0.0, s1))
    fst.add_arc(s1, kaldifst.StdArc(2, 2, 0.0, s2))
    fst.set_final(s2, 0.0)
    return fst


def labels(best):
    """Walk the linear best path returning [(ilabel, olabel), ...].

    to_str() layout is openfst's FstPrinter: "src dest ilabel olabel [w]"
    (weight omitted when One; LatticeWeight prints "v1,v2").  The walk is
    bounded by num_states so a malformed FST cannot loop forever.
    """
    raw = [ln.split() for ln in best.to_str().strip().splitlines()]
    finals = {int(f[0]) for f in raw if len(f) in (1, 2)}
    by_src = {}
    for f in raw:
        if len(f) in (4, 5):
            src, dest, ilabel, olabel = map(int, f[:4])
            assert src not in by_src, f"duplicate arc from state {src}"
            by_src[src] = (dest, ilabel, olabel)
    out = []
    state = best.start
    for _ in range(best.num_states + 1):
        if state in finals:
            return out
        assert state in by_src, f"no arc from state {state}"
        dest, ilabel, olabel = by_src[state]
        out.append((ilabel, olabel))
        state = dest
    raise AssertionError(f"best path does not reach a final state:\n{best.to_str()}")


class TestDecoderKeepsGraphAlive(unittest.TestCase):
    def _check(self, decoder):
        dec = TableDecodable()
        decoder.decode(dec)   # FasterDecoder returns void (kaldi signature)
        ok, best = decoder.get_best_path()
        self.assertTrue(ok)
        self.assertEqual(labels(best), [(1, 1), (2, 2)])

    def test_faster_decoder_temporary_graph(self):
        dec = FasterDecoder(build_graph(), FasterDecoderOptions(beam=16.0))
        gc.collect()  # the temporary graph must be kept alive by the decoder
        self._check(dec)

    def test_lattice_faster_decoder_temporary_graph(self):
        dec = LatticeFasterDecoder(build_graph(), LatticeFasterDecoderConfig())
        gc.collect()
        self._check(dec)

    def test_lattice_simple_decoder_temporary_graph(self):
        dec = LatticeSimpleDecoder(build_graph(), LatticeSimpleDecoderConfig())
        gc.collect()
        self._check(dec)

    def test_simple_decoder_temporary_graph(self):
        dec = TableDecodable()
        graph = build_graph()
        decoder = SimpleDecoder(graph, 16.0)
        del graph
        gc.collect()
        self._check(decoder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
