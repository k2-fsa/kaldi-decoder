#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_fst_interop_py
"""Zero-copy FST interop between the kaldifst python module and kaldi_decoder.

FSTs cross the module boundary BY POINTER via kaldifst's
`_vector_fst_addr/_borrow/_take` helpers (see python/csrc/fst-caster.h):
- inputs are borrowed (the decoder never takes ownership or frees them),
- outputs are handed over as owning kaldifst python objects,
- a returned FST must be re-usable as an input to another decoder.
"""
import unittest

try:
    import kaldifst
except ImportError:
    raise SystemExit(77)  # optional dependency missing -> ctest "skipped"

from kaldi_decoder import (
    DecodableInterface,
    FasterDecoder,
    FasterDecoderOptions,
    LatticeSimpleDecoder,
    LatticeSimpleDecoderConfig,
    SimpleDecoder,
)


class FakeDecodable(DecodableInterface):
    """Deterministic scores: the correct label at frame t is t+1."""

    def __init__(self, num_frames=2, num_labels=2):
        super().__init__()
        self.num_frames = num_frames
        self.num_labels = num_labels

    def num_frames_ready(self):
        return self.num_frames

    def is_last_frame(self, frame):
        return frame == self.num_frames - 1

    def num_indices(self):
        return self.num_labels

    def log_likelihood(self, frame, index):
        return 0.0 if index == frame + 1 else -10.0


def build_linear_graph():
    fst = kaldifst.StdVectorFst()
    s0 = fst.add_state()
    s1 = fst.add_state()
    s2 = fst.add_state()
    fst.start = s0
    fst.add_arc(s0, kaldifst.StdArc(1, 1, 0.0, s1))
    fst.add_arc(s1, kaldifst.StdArc(2, 2, 0.0, s2))
    fst.set_final(s2, 0.0)
    return fst


def best_path_labels(best_path):
    """Walk the linear best path returning [(ilabel, olabel), ...].

    to_str() layout is openfst's FstPrinter: "src dest ilabel olabel [w]"
    (weight omitted when One; LatticeWeight prints "v1,v2").  The walk is
    bounded by num_states so a malformed FST cannot loop forever.
    """
    raw = [ln.split() for ln in best_path.to_str().strip().splitlines()]
    finals = {int(f[0]) for f in raw if len(f) in (1, 2)}
    by_src = {}
    for f in raw:
        if len(f) in (4, 5):
            src, dest, ilabel, olabel = map(int, f[:4])
            assert src not in by_src, f"duplicate arc from state {src}"
            by_src[src] = (dest, ilabel, olabel)
    out = []
    state = best_path.start
    for _ in range(best_path.num_states + 1):
        if state in finals:
            return out
        assert state in by_src, f"no arc from state {state}"
        dest, ilabel, olabel = by_src[state]
        out.append((ilabel, olabel))
        state = dest
    raise AssertionError(f"best path does not reach a final state:\n{best_path.to_str()}")


class TestFstInterop(unittest.TestCase):
    def test_borrowed_input_not_owned(self):
        graph = build_linear_graph()
        before = graph.to_str()
        d = FakeDecodable()
        decoder = FasterDecoder(graph, FasterDecoderOptions(beam=16.0))
        decoder.init_decoding()
        decoder.advance_decoding(d)
        # The decoder borrowed the graph; it must be untouched and usable.
        self.assertEqual(graph.to_str(), before)

    def test_same_graph_two_decoders(self):
        # Borrowing must not transfer ownership: the same python object feeds
        # two live decoders at once.
        graph = build_linear_graph()
        d = FakeDecodable()
        dec1 = FasterDecoder(graph, FasterDecoderOptions(beam=16.0))
        dec2 = FasterDecoder(graph, FasterDecoderOptions(beam=16.0))
        for dec in (dec1, dec2):
            dec.init_decoding()
            dec.advance_decoding(d)
            ok, best = dec.get_best_path()
            self.assertTrue(ok)
            self.assertEqual(best_path_labels(best), [(1, 1), (2, 2)])

    def test_returned_lattice_is_kaldifst_object(self):
        # get_best_path returns an owning kaldifst python object (Lattice for
        # LatticeArc results, StdVectorFst for StdArc ones) — never a foreign
        # capsule; the caller owns it and can introspect it freely.
        graph = build_linear_graph()
        d = FakeDecodable()
        decoder = FasterDecoder(graph, FasterDecoderOptions(beam=16.0))
        decoder.init_decoding()
        decoder.advance_decoding(d)
        ok, best = decoder.get_best_path()
        self.assertTrue(ok)
        self.assertIsInstance(best, kaldifst.Lattice)
        self.assertEqual(best.num_states, 3)
        self.assertGreaterEqual(best.num_arcs(best.start), 1)

    def test_simple_decoder(self):
        graph = build_linear_graph()
        d = FakeDecodable()
        dec = SimpleDecoder(graph, beam=16.0)
        self.assertTrue(dec.decode(d))
        self.assertTrue(dec.reached_final())
        ok, best = dec.get_best_path()
        self.assertTrue(ok)
        self.assertEqual(best_path_labels(best), [(1, 1), (2, 2)])

    def test_lattice_simple_decoder(self):
        graph = build_linear_graph()
        d = FakeDecodable()
        dec = LatticeSimpleDecoder(graph, LatticeSimpleDecoderConfig())
        self.assertTrue(dec.decode(d))
        ok, best = dec.get_best_path()
        self.assertTrue(ok)
        self.assertIsInstance(best, kaldifst.Lattice)
        # Lattice to_str() prints "src dest ilabel olabel" (weight omitted
        # when One); walk start -> final and collect the input labels.
        arcs = {}
        finals = set()
        for line in best.to_str().splitlines():
            f = line.split()
            if len(f) == 4:
                src, dest, ilabel, olabel = map(int, f)
                arcs[src] = (dest, ilabel, olabel)
            elif len(f) == 1:
                finals.add(int(f[0]))
        labels = []
        state = best.start
        while state not in finals:
            state, ilabel, olabel = arcs[state]
            labels.append((ilabel, olabel))
        self.assertEqual(labels, [(1, 1), (2, 2)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
