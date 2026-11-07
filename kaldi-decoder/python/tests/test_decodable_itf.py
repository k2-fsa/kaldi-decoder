#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_decodable_itf_py
"""DecodableInterface trampoline + DecodableCtc sanity checks.

A Python subclass of DecodableInterface must be usable as the decodable of
every decoder (the trampoline keeps virtual dispatch on the Python side).
"""
import unittest

try:
    import kaldifst  # noqa: F401
except ImportError:
    raise SystemExit(77)  # optional dependency missing -> ctest "skipped"

from kaldi_decoder import (DecodableCtc, DecodableInterface, DecodableMatrixMapped,
                         DecodableMatrixMappedOffset, DecodableMatrixScaled,
                         FasterDecoder, FasterDecoderOptions, TransitionInformation)

class FakeTm(TransitionInformation):
    """Minimal 1-based tid -> pdf map."""

    def __init__(self):
        super().__init__()
        self.tid_to_pdf = [-1, 0, 1, 1, 2, 3]

    def transition_ids_equivalent(self, a, b):
        return self.tid_to_pdf[a] == self.tid_to_pdf[b]

    def transition_id_is_start_of_phone(self, tid):
        return True

    def transition_id_to_phone(self, tid):
        return 1

    def is_final(self, tid):
        return False

    def is_self_loop(self, tid):
        return False

    def transition_id_to_pdf_array(self):
        return self.tid_to_pdf

    def num_pdfs(self):
        return 4


NUM_LABELS = 2
NUM_FRAMES = 2


def build_linear_graph() -> "kaldifst.StdVectorFst":
    import kaldifst

    fst = kaldifst.StdVectorFst()
    s0 = fst.add_state()
    s1 = fst.add_state()
    s2 = fst.add_state()
    fst.start = s0
    fst.add_arc(s0, kaldifst.StdArc(1, 1, 0.0, s1))
    fst.add_arc(s1, kaldifst.StdArc(2, 2, 0.0, s2))
    fst.set_final(s2, 0.0)
    return fst


class FakeDecodable(DecodableInterface):
    """Deterministic scores: the correct label at frame t is t+1."""

    def num_frames_ready(self):
        return NUM_FRAMES

    def is_last_frame(self, frame):
        return frame == NUM_FRAMES - 1

    def num_indices(self):
        return NUM_LABELS

    def log_likelihood(self, frame, index):
        return 0.0 if index == frame + 1 else -10.0


class TestDecodableItf(unittest.TestCase):
    def test_trampoline_scores(self):
        d = FakeDecodable()
        self.assertEqual(d.num_frames_ready(), NUM_FRAMES)
        self.assertTrue(d.is_last_frame(1))
        self.assertFalse(d.is_last_frame(0))
        self.assertEqual(d.num_indices(), NUM_LABELS)
        self.assertEqual(d.log_likelihood(0, 1), 0.0)
        self.assertEqual(d.log_likelihood(0, 2), -10.0)

    def test_decode_with_python_decodable(self):
        graph = build_linear_graph()
        decodable = FakeDecodable()
        decoder = FasterDecoder(graph, FasterDecoderOptions(beam=16.0))
        decoder.init_decoding()
        decoder.advance_decoding(decodable)
        self.assertEqual(decoder.num_frames_decoded(), NUM_FRAMES)
        self.assertTrue(decoder.reached_final())
        ok, best_path = decoder.get_best_path()
        self.assertTrue(ok)
        # to_str() layout: "src dest ilabel olabel [w]" (openfst FstPrinter)
        labels = []
        for line in best_path.to_str().splitlines():
            f = line.split()
            if len(f) in (4, 5):
                labels.append((int(f[2]), int(f[3])))  # (ilabel, olabel)
        self.assertEqual(labels, [(1, 1), (2, 2)])

    def test_decodable_ctc(self):
        # 2 frames x 2 classes log-probs; argmax is 1 then 2.
        import numpy as np

        loglikes = np.array([[-0.1, -5.0], [-5.0, -0.1]], dtype=np.float32)
        d = DecodableCtc(loglikes)
        self.assertEqual(d.num_frames_ready(), 2)
        self.assertEqual(d.num_indices(), 2)
        self.assertAlmostEqual(d.log_likelihood(0, 1), -0.1, places=5)
        self.assertAlmostEqual(d.log_likelihood(1, 2), -0.1, places=5)


class TestUncoveredAccessors(unittest.TestCase):
    """Coverage-driven: the remaining untested accessors/error paths."""

    def test_num_frames_ready_default_throws(self):
        # The base-class default raises "not implemented" (kaldi semantics).
        class Minimal(DecodableInterface):
            def __init__(self):
                super().__init__()

            def log_likelihood(self, frame, index):
                return 0.0

            def is_last_frame(self, frame):
                return True

            def num_indices(self):
                return 1

        with self.assertRaises(RuntimeError):
            Minimal().num_frames_ready()

    def test_decodable_ctc_is_last_frame(self):
        import numpy as np

        d = DecodableCtc(np.zeros((3, 2), dtype=np.float32))
        self.assertTrue(d.is_last_frame(2))
        self.assertFalse(d.is_last_frame(1))

    def test_matrix_decodable_accessors(self):
        import numpy as np

        tm = FakeTm()
        likes = np.zeros((2, 4), dtype=np.float32)
        mapped = DecodableMatrixMapped(tm, likes, 0)
        self.assertEqual(mapped.num_indices(), 5)
        off = DecodableMatrixMappedOffset(tm)
        self.assertEqual(off.num_indices(), 5)
        scaled = DecodableMatrixScaled(likes, 1.0)
        self.assertTrue(scaled.is_last_frame(1))
        self.assertFalse(scaled.is_last_frame(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
