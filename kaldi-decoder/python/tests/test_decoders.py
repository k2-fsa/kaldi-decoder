#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_decoders_py
"""Decoder golden scenarios through the Python bindings.

Mirrors the C++ gtests (simple-decoder-test.cc, faster-decoder-test.cc,
lattice-simple-decoder-test.cc): tiny graphs with hand-computed best paths,
re-verified through the pybind layer.
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
    LatticeFasterDecoder,
    LatticeFasterDecoderConfig,
    LatticeSimpleDecoder,
    LatticeSimpleDecoderConfig,
    SimpleDecoder,
)


class TableDecodable(DecodableInterface):
    """score table [frame][label], labels 1-based."""

    def __init__(self, scores, num_indices):
        super().__init__()
        self.scores = scores
        self.n = num_indices

    def num_frames_ready(self):
        return len(self.scores)

    def is_last_frame(self, frame):
        return frame == len(self.scores) - 1

    def num_indices(self):
        return self.n

    def log_likelihood(self, frame, index):
        return self.scores[frame][index - 1]


def make_linear_graph(with_final=True):
    fst = kaldifst.StdVectorFst()
    s0 = fst.add_state()
    s1 = fst.add_state()
    s2 = fst.add_state()
    fst.start = s0
    fst.add_arc(s0, kaldifst.StdArc(1, 1, 0.0, s1))
    fst.add_arc(s1, kaldifst.StdArc(2, 2, 0.0, s2))
    if with_final:
        fst.set_final(s2, 0.0)
    return fst


def make_two_branch_graph(w_a, w_b):
    fst = kaldifst.StdVectorFst()
    s0 = fst.add_state()
    s1 = fst.add_state()
    s2 = fst.add_state()
    sf = fst.add_state()
    fst.start = s0
    fst.add_arc(s0, kaldifst.StdArc(1, 1, w_a, s1))
    fst.add_arc(s1, kaldifst.StdArc(2, 2, w_a, sf))
    fst.add_arc(s0, kaldifst.StdArc(3, 3, w_b, s2))
    fst.add_arc(s2, kaldifst.StdArc(4, 4, w_b, sf))
    fst.set_final(sf, 0.0)
    return fst


def _walk_labels(best):
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

def branch_a_scores(gap):
    return [[0.0, -gap, -gap, -gap], [-gap, 0.0, -gap, -gap]]


def branch_b_scores(gap):
    return [[-gap, -gap, 0.0, -gap], [-gap, -gap, -gap, 0.0]]


class TestDecoders(unittest.TestCase):
    def test_simple_decoder_linear(self):
        # The decoder borrows the graph; it must stay referenced in a local
        # for the decoder's whole lifetime (same for the decodable).
        graph = make_linear_graph()
        dec = TableDecodable(branch_a_scores(10), 4)
        decoder = SimpleDecoder(graph, 16.0)
        self.assertTrue(decoder.decode(dec))
        self.assertTrue(decoder.reached_final())
        ok, best = decoder.get_best_path()
        self.assertTrue(ok)
        self.assertEqual(_walk_labels(best), [(1, 1), (2, 2)])

    def test_faster_decoder_linear(self):
        graph = make_linear_graph()
        dec = TableDecodable(branch_a_scores(10), 4)
        decoder = FasterDecoder(graph, FasterDecoderOptions(beam=16.0))
        decoder.init_decoding()
        decoder.advance_decoding(dec)
        self.assertEqual(decoder.num_frames_decoded(), 2)
        self.assertTrue(decoder.reached_final())
        ok, best = decoder.get_best_path()
        self.assertTrue(ok)
        self.assertEqual(_walk_labels(best), [(1, 1), (2, 2)])

    def test_lattice_simple_decoder_linear(self):
        graph = make_linear_graph()
        dec = TableDecodable(branch_a_scores(10), 4)
        decoder = LatticeSimpleDecoder(graph, LatticeSimpleDecoderConfig())
        self.assertTrue(decoder.decode(dec))
        self.assertTrue(decoder.reached_final())
        ok, best = decoder.get_best_path()
        self.assertTrue(ok)
        self.assertEqual(_walk_labels(best), [(1, 1), (2, 2)])

    def test_two_branches_acoustic(self):
        for cls, kwargs in (
            (SimpleDecoder, {"beam": 16.0}),
            (FasterDecoder, {}),
        ):
            graph = make_two_branch_graph(0.0, 0.0)
            if cls is SimpleDecoder:
                make = lambda: SimpleDecoder(graph, 16.0)  # noqa: E731
            else:
                make = lambda: FasterDecoder(  # noqa: E731
                    graph, FasterDecoderOptions(beam=16.0)
                )
            for scores, expected in (
                (branch_a_scores(10), [(1, 1), (2, 2)]),
                (branch_b_scores(10), [(3, 3), (4, 4)]),
            ):
                dec = TableDecodable(scores, 4)
                decoder = make()
                if cls is FasterDecoder:
                    decoder.init_decoding()
                    decoder.advance_decoding(dec)
                else:
                    self.assertTrue(decoder.decode(dec))
                ok, best = decoder.get_best_path()
                self.assertTrue(ok, f"{cls.__name__}")
                self.assertEqual(_walk_labels(best), expected, cls.__name__)

    def test_two_branches_acoustic_lattice(self):
        graph = make_two_branch_graph(0.0, 0.0)
        cfg = LatticeSimpleDecoderConfig(beam=16.0, lattice_beam=10.0)
        for scores, expected in (
            (branch_a_scores(10), [(1, 1), (2, 2)]),
            (branch_b_scores(10), [(3, 3), (4, 4)]),
        ):
            dec = TableDecodable(scores, 4)
            decoder = LatticeSimpleDecoder(graph, cfg)
            self.assertTrue(decoder.decode(dec))
            ok, best = decoder.get_best_path()
            self.assertTrue(ok)
            self.assertEqual(_walk_labels(best), expected)

    def test_two_branches_graph_weight(self):
        uniform = TableDecodable([[0.0] * 4, [0.0] * 4], 4)
        for w_a, w_b, expected in ((0.5, 2.0, [(1, 1), (2, 2)]), (2.0, 0.5, [(3, 3), (4, 4)])):
            graph = make_two_branch_graph(w_a, w_b)
            dec = TableDecodable([[0.0] * 4, [0.0] * 4], 4)
            decoder = SimpleDecoder(graph, 16.0)
            self.assertTrue(decoder.decode(dec))
            ok, best = decoder.get_best_path()
            self.assertTrue(ok)
            self.assertEqual(_walk_labels(best), expected)
        del uniform

    def test_lattice_faster_decoder_linear(self):
        graph = make_linear_graph()
        dec = TableDecodable(branch_a_scores(10), 4)
        decoder = LatticeFasterDecoder(graph, LatticeFasterDecoderConfig())
        self.assertTrue(decoder.decode(dec))
        self.assertEqual(decoder.num_frames_decoded(), 2)
        self.assertTrue(decoder.reached_final())
        ok, best = decoder.get_best_path()
        self.assertTrue(ok)
        self.assertEqual(_walk_labels(best), [(1, 1), (2, 2)])

    def test_lattice_faster_decoder_two_branches(self):
        graph = make_two_branch_graph(0.0, 0.0)
        cfg = LatticeFasterDecoderConfig(beam=16.0, lattice_beam=10.0)
        for scores, expected in (
            (branch_a_scores(10), [(1, 1), (2, 2)]),
            (branch_b_scores(10), [(3, 3), (4, 4)]),
        ):
            dec = TableDecodable(scores, 4)
            decoder = LatticeFasterDecoder(graph, cfg)
            self.assertTrue(decoder.decode(dec))
            ok, best = decoder.get_best_path()
            self.assertTrue(ok)
            self.assertEqual(_walk_labels(best), expected)

    def test_lattice_faster_decoder_raw_lattice(self):
        graph = make_two_branch_graph(0.0, 0.0)
        cfg = LatticeFasterDecoderConfig(beam=16.0, lattice_beam=10.0)
        dec = TableDecodable(branch_a_scores(1), 4)  # soft gap keeps both
        decoder = LatticeFasterDecoder(graph, cfg)
        self.assertTrue(decoder.decode(dec))
        ok, raw = decoder.get_raw_lattice()
        self.assertTrue(ok)
        labels = set()
        for line in raw.to_str().splitlines():
            f = line.split()
            if len(f) >= 4:
                labels.add(int(f[2]))  # lattice layout: "src dest ilabel ..."
        self.assertEqual(labels, {1, 2, 3, 4})

    def test_decodable_none_raises(self):
        # pybind turns None into nullptr for raw-pointer params; the
        # bindings use references so None must raise TypeError, not crash.
        graph = make_two_branch_graph(0.0, 0.0)
        dec = TableDecodable([[0.0, -10.0, -10.0, -10.0]], 4)
        for decoder in (
            SimpleDecoder(graph, 16.0),
            FasterDecoder(graph, FasterDecoderOptions(beam=16.0)),
            LatticeSimpleDecoder(graph, LatticeSimpleDecoderConfig()),
            LatticeFasterDecoder(graph, LatticeFasterDecoderConfig()),
        ):
            with self.assertRaises(TypeError):
                decoder.decode(None)

    def test_no_final_state(self):
        graph = make_linear_graph(with_final=False)
        dec = TableDecodable([[0.0, -10.0], [-10.0, 0.0]], 2)
        decoder = SimpleDecoder(graph, 16.0)
        decoder.decode(dec)
        self.assertFalse(decoder.reached_final())
        # kaldi semantics: best PARTIAL path + true (SimpleDecoder::GetBestPath)
        ok, best = decoder.get_best_path()
        self.assertTrue(ok)
        self.assertEqual(_walk_labels(best), [(1, 1), (2, 2)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
