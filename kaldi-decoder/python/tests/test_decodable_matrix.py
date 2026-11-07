#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_decodable_matrix_py
"""Matrix-backed decodables + decoder wrappers through the Python bindings.

Mirrors decodable-matrix-test.cc and decoder-wrappers-test.cc, and exercises
the TransitionInformation trampoline (a python subclass providing the
tid -> pdf map).
"""
import gc
import unittest

try:
    import kaldifst
    import numpy as np
except ImportError:
    raise SystemExit(77)  # optional dependency missing -> ctest "skipped"

from kaldi_decoder import (
    DecodableCtc as kd_ctc,
    AlignConfig,
    DecodableInterface,
    DecodableMatrixMapped,
    DecodableMatrixMappedOffset,
    DecodableMatrixScaled,
    DecodableMatrixScaledMapped,
    LatticeFasterDecoder,
    LatticeFasterDecoderConfig,
    LatticeSimpleDecoder,
    LatticeSimpleDecoderConfig,
    TransitionInformation,
    align_utterance_wrapper,
    lattice_best_path,
    decode_utterance_lattice_faster,
    decode_utterance_lattice_simple,
)


class FakeTm(TransitionInformation):
    """1-based tid -> pdf map: tids 1..5 map to pdfs 0,1,1,2,3."""

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


class TableDecodable(DecodableInterface):
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


def make_linear_word_graph():
    """s0 --1/10--> s1 --2/20--> s2 (final)."""
    fst = kaldifst.StdVectorFst()
    s0 = fst.add_state()
    s1 = fst.add_state()
    s2 = fst.add_state()
    fst.start = s0
    fst.add_arc(s0, kaldifst.StdArc(1, 10, 0.0, s1))
    fst.add_arc(s1, kaldifst.StdArc(2, 20, 0.0, s2))
    fst.set_final(s2, 0.0)
    return fst


class TestTransitionInformationTrampoline(unittest.TestCase):
    def test_accessors(self):
        tm = FakeTm()
        self.assertEqual(tm.num_pdfs(), 4)
        self.assertEqual(tm.num_transition_ids(), 5)
        self.assertEqual(list(tm.transition_id_to_pdf_array()), [-1, 0, 1, 1, 2, 3])
        self.assertTrue(tm.transition_ids_equivalent(2, 3))
        self.assertFalse(tm.transition_ids_equivalent(2, 4))


class TestDecodableMatrix(unittest.TestCase):
    def test_scaled_mapped(self):
        tm = FakeTm()
        likes = np.array(
            [[0.0, -1.0, -2.0, -3.0], [-4.0, -5.0, -6.0, -7.0]], dtype=np.float32
        )
        dec = DecodableMatrixScaledMapped(tm, likes, 2.0)
        self.assertEqual(dec.num_frames_ready(), 2)
        self.assertEqual(dec.num_indices(), 5)
        self.assertTrue(dec.is_last_frame(1))
        self.assertAlmostEqual(dec.log_likelihood(0, 2), 2.0 * -1.0, places=5)
        self.assertAlmostEqual(dec.log_likelihood(1, 3), 2.0 * -5.0, places=5)
        self.assertAlmostEqual(dec.log_likelihood(1, 5), 2.0 * -7.0, places=5)

    def test_mapped_frame_offset(self):
        tm = FakeTm()
        likes = np.array([[-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
        dec = DecodableMatrixMapped(tm, likes, 3)
        self.assertEqual(dec.num_frames_ready(), 4)
        self.assertTrue(dec.is_last_frame(3))
        self.assertAlmostEqual(dec.log_likelihood(3, 1), -1.0, places=5)

    def test_mapped_offset_chunks(self):
        tm = FakeTm()
        dec = DecodableMatrixMappedOffset(tm)
        chunk1 = np.array(
            [[0.0, -1.0, -2.0, -3.0], [-4.0, -5.0, -6.0, -7.0]], dtype=np.float32
        )
        dec.accept_loglikes(chunk1, 0)
        self.assertEqual(dec.num_frames_ready(), 2)
        self.assertAlmostEqual(dec.log_likelihood(1, 5), -7.0, places=5)
        chunk2 = np.array([[-8.0, -9.0, -10.0, -11.0]], dtype=np.float32)
        dec.accept_loglikes(chunk2, 1)
        self.assertEqual(dec.num_frames_ready(), 3)
        self.assertEqual(dec.first_available_frame(), 1)
        self.assertAlmostEqual(dec.log_likelihood(1, 1), -4.0, places=5)
        self.assertAlmostEqual(dec.log_likelihood(2, 1), -8.0, places=5)
        dec.input_is_finished()
        self.assertTrue(dec.is_last_frame(2))

    def test_scaled(self):
        likes = np.array([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]], dtype=np.float32)
        dec = DecodableMatrixScaled(likes, 0.5)
        self.assertEqual(dec.num_frames_ready(), 2)
        self.assertEqual(dec.num_indices(), 3)
        self.assertAlmostEqual(dec.log_likelihood(1, 3), 0.5 * -5.0, places=5)

    def test_lifetime_after_numpy_gone(self):
        # The decodable owns a copy: dropping the numpy array must not break
        # lookups (regression for use-after-free).
        tm = FakeTm()
        likes = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        dec = DecodableMatrixScaledMapped(tm, likes, 1.0)
        del likes
        self.assertAlmostEqual(dec.log_likelihood(0, 1), 1.0, places=5)


class TestBindingRegressions(unittest.TestCase):
    def test_decodable_keeps_tm_alive(self):
        # keep_alive<1,2>: the decodable must hold the transition model.
        # Delete the tm (and drop every other reference) then USE the
        # decodable -- without keep_alive this is a use-after-free.
        likes = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        for make in (
            lambda t: DecodableMatrixScaledMapped(t, likes, 1.0),
            lambda t: DecodableMatrixMapped(t, likes, 0),
            lambda t: DecodableMatrixMappedOffset(t),
        ):
            tm = FakeTm()
            dec = make(tm)
            del tm
            gc.collect()
            if isinstance(dec, DecodableMatrixMappedOffset):
                dec.accept_loglikes(likes, 0)
            self.assertAlmostEqual(dec.log_likelihood(0, 1), 1.0, places=5)

    def test_frame_out_of_range_raises(self):
        tm = FakeTm()
        likes = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        for dec in (
            DecodableMatrixScaledMapped(tm, likes, 1.0),
            DecodableMatrixMapped(tm, likes, 0),
            DecodableMatrixMappedOffset(tm),
        ):
            with self.assertRaises(RuntimeError):
                dec.log_likelihood(999, 1)
            with self.assertRaises(RuntimeError):
                dec.log_likelihood(-1, 1)

    def test_decodable_ctc_empty_and_bounds(self):
        # Regression: empty-matrix ctor used to be UB; queries must raise.
        empty = kd_ctc(np.zeros((0, 5), dtype=np.float32))
        self.assertEqual(empty.num_frames_ready(), 0)
        with self.assertRaises(RuntimeError):
            empty.log_likelihood(0, 1)
        dec = kd_ctc(np.zeros((2, 3), dtype=np.float32))
        with self.assertRaises(RuntimeError):
            dec.log_likelihood(2, 1)   # frame OOB
        with self.assertRaises(RuntimeError):
            dec.log_likelihood(0, 0)   # index OOB (1-based)

    def test_throw_paths_repeatable(self):
        # Wrong column count must raise cleanly every time (the owned matrix
        # copy is freed on the exception path -- unique_ptr).  Leak detection
        # itself belongs to the ASan/LSan sanitizer run, not this test.
        tm = FakeTm()
        bad = np.zeros((4, 3), dtype=np.float32)  # 3 cols != 4 pdfs
        for _ in range(200):
            with self.assertRaises(RuntimeError):
                DecodableMatrixScaledMapped(tm, bad, 1.0)
        dec = DecodableMatrixMappedOffset(tm)
        for _ in range(200):
            with self.assertRaises(RuntimeError):
                dec.accept_loglikes(bad, 0)
        # still usable afterwards
        good = np.zeros((4, 4), dtype=np.float32)
        dec.accept_loglikes(good, 0)
        self.assertEqual(dec.num_frames_ready(), 4)


class TestDecoderWrappers(unittest.TestCase):
    def test_align_utterance_wrapper(self):
        graph = make_linear_word_graph()
        dec = TableDecodable([[0.0, -10.0], [-10.0, 0.0]], 2)
        (
            num_done,
            num_error,
            num_retried,
            tot_like,
            frame_count,
            alignment,
            words,
        ) = align_utterance_wrapper(
            config=AlignConfig(),
            utt="utt",
            acoustic_scale=1.0,
            fst=graph,
            decodable=dec,
            num_done=0,
            num_error=0,
            num_retried=0,
            tot_like=0.0,
            frame_count=0,
        )
        self.assertEqual(num_done, 1)
        self.assertEqual(num_error, 0)
        self.assertEqual(alignment, [1, 2])
        self.assertEqual(words, [10, 20])
        self.assertEqual(frame_count, 2)

    def test_decode_utterance_lattice_faster(self):
        graph = make_linear_word_graph()
        dec = TableDecodable([[0.0, -10.0], [-10.0, 0.0]], 2)
        tm = FakeTm()
        decoder = LatticeFasterDecoder(graph, LatticeFasterDecoderConfig())
        ok, ali, words, log_like, lat = decode_utterance_lattice_faster(
            decoder=decoder,
            decodable=dec,
            trans_model=tm,
            utt="utt",
            allow_partial=False,
            acoustic_scale=1.0,
        )
        self.assertTrue(ok)
        self.assertEqual(ali, [1, 2])
        self.assertEqual(words, [10, 20])
        self.assertAlmostEqual(log_like, 0.0, places=4)

    def test_lattice_best_path_scales(self):
        import kaldifst

        lat = kaldifst.Lattice()
        s0 = lat.add_state()
        sa = lat.add_state()
        sb = lat.add_state()
        sf = lat.add_state()
        lat.start = s0
        lat.add_arc(s0, kaldifst.LatticeArc(1, 10, kaldifst.LatticeWeight(2.0, 0.0), sa))
        lat.add_arc(sa, kaldifst.LatticeArc(2, 20, kaldifst.LatticeWeight(0.0, 0.0), sf))
        lat.add_arc(s0, kaldifst.LatticeArc(3, 30, kaldifst.LatticeWeight(0.0, 0.0), sb))
        lat.add_arc(sb, kaldifst.LatticeArc(4, 40, kaldifst.LatticeWeight(0.0, 2.0), sf))
        lat.set_final(sf, kaldifst.LatticeWeight(0.0, 0.0))

        ok, ali, words, like = lattice_best_path(lat, lm_scale=1.0, acoustic_scale=1.0)
        self.assertTrue(ok)
        self.assertEqual(ali, [3, 4])  # tie broken by smaller graph cost
        self.assertEqual(words, [30, 40])

        ok, ali, words, like = lattice_best_path(lat, lm_scale=1.0, acoustic_scale=2.0)
        self.assertTrue(ok)
        self.assertEqual(ali, [1, 2])
        self.assertEqual(words, [10, 20])
        self.assertAlmostEqual(like, -2.0, places=4)

    def test_lattice_best_path_soft_fail(self):
        # Regression: an empty lattice used to abort; now returns ok=False
        # (kaldi's lattice-best-path skips bad lattices in a sweep).
        import kaldifst

        empty = kaldifst.Lattice()
        empty.add_state()
        ok, ali, words, like = lattice_best_path(empty)
        self.assertFalse(ok)
        self.assertEqual(ali, [])
        self.assertEqual(words, [])

    def test_decode_utterance_lattice_simple(self):
        graph = make_linear_word_graph()
        dec = TableDecodable([[0.0, -10.0], [-10.0, 0.0]], 2)
        tm = FakeTm()
        decoder = LatticeSimpleDecoder(graph, LatticeSimpleDecoderConfig())
        ok, ali, words, log_like, lat = decode_utterance_lattice_simple(
            decoder=decoder,
            decodable=dec,
            trans_model=tm,
            utt="utt",
            allow_partial=False,
            acoustic_scale=1.0,
        )
        self.assertTrue(ok)
        self.assertEqual(ali, [1, 2])
        self.assertEqual(words, [10, 20])


if __name__ == "__main__":
    unittest.main(verbosity=2)
