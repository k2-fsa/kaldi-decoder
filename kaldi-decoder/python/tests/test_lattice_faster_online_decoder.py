#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_lattice_faster_online_decoder_py
"""LatticeFasterOnlineDecoder through the Python bindings: the backpointer
traceback (get_best_path_transition_ids) must equal the best path of the raw
lattice at every frame of an incremental decode."""
import random
import unittest

try:
    import kaldifst
except ImportError:
    raise SystemExit(77)  # optional dependency missing -> ctest "skipped"

from kaldi_decoder import (
    DecodableInterface,
    LatticeFasterDecoderConfig,
    LatticeFasterOnlineDecoder,
    lattice_best_path,
)


class TableDecodable(DecodableInterface):
    """score table [frame][label], labels 1-based; frames become ready as
    `ready` is raised (incremental decoding)."""

    def __init__(self, scores, num_indices):
        super().__init__()
        self.scores = scores
        self.n = num_indices
        self.ready = len(scores)

    def num_frames_ready(self):
        return self.ready

    def is_last_frame(self, frame):
        return frame == len(self.scores) - 1

    def num_indices(self):
        return self.n

    def log_likelihood(self, frame, index):
        return self.scores[frame][index - 1]


def make_random_graph(num_states, num_labels, rng):
    fst = kaldifst.StdVectorFst()
    for _ in range(num_states):
        fst.add_state()
    fst.start = 0
    for s in range(num_states):
        fst.add_arc(
            s, kaldifst.StdArc(rng.randint(1, num_labels), 0, rng.uniform(0, 2), s)
        )
        for _ in range(3):
            label = rng.randint(1, num_labels)
            fst.add_arc(
                s,
                kaldifst.StdArc(
                    label, label, rng.uniform(0, 2), rng.randrange(num_states)
                ),
            )
        if s + 1 < num_states:
            fst.add_arc(s, kaldifst.StdArc(0, 0, rng.uniform(0, 2), s + 1))
        if s % 3 == 2:
            fst.set_final(s, rng.uniform(0, 2))
    return fst


class TestLatticeFasterOnlineDecoder(unittest.TestCase):
    def test_traceback_matches_raw_lattice(self):
        rng = random.Random(20261127)
        num_labels = 6
        for trial in range(10):
            graph = make_random_graph(8, num_labels, rng)
            scores = [
                [rng.uniform(-8, 0) for _ in range(num_labels)] for _ in range(20)
            ]
            dec = TableDecodable(scores, num_labels)
            config = LatticeFasterDecoderConfig(beam=12.0, lattice_beam=6.0)
            decoder = LatticeFasterOnlineDecoder(graph, config)
            decoder.init_decoding()
            for n in range(1, len(scores) + 1):
                dec.ready = n
                decoder.advance_decoding(dec)
                ids = decoder.get_best_path_transition_ids(use_final_probs=False)
                self.assertEqual(len(ids), n)
                ok, raw = decoder.get_raw_lattice(use_final_probs=False)
                self.assertTrue(ok)
                ok, alignment, _, _ = lattice_best_path(raw)
                self.assertTrue(ok)
                self.assertEqual(ids, [a for a in alignment if a != 0], (trial, n))
            decoder.finalize_decoding()
            ids = decoder.get_best_path_transition_ids(use_final_probs=True)
            ok, raw = decoder.get_raw_lattice(use_final_probs=True)
            _, alignment, _, _ = lattice_best_path(raw)
            self.assertEqual(ids, [a for a in alignment if a != 0])
            with self.assertRaises(RuntimeError):
                decoder.get_best_path_transition_ids(use_final_probs=False)

    def test_no_frames(self):
        graph = make_random_graph(4, 3, random.Random(1))
        decoder = LatticeFasterOnlineDecoder(graph, LatticeFasterDecoderConfig())
        decoder.init_decoding()
        self.assertEqual(decoder.get_best_path_transition_ids(), [])


if __name__ == "__main__":
    unittest.main()
