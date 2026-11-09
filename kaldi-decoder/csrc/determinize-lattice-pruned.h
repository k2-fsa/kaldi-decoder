// kaldi-decoder/csrc/determinize-lattice-pruned.h

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
// Copyright (c)  2023-2026  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/lat/determinize-lattice-pruned.h

#ifndef KALDI_DECODER_CSRC_DETERMINIZE_LATTICE_PRUNED_H_
#define KALDI_DECODER_CSRC_DETERMINIZE_LATTICE_PRUNED_H_

#include <sstream>
#include <string>

#include "fst/fst.h"
#include "fst/fstlib.h"

namespace kaldi_decoder {

/// @brief Options for lattice determinization (kaldi's
/// DeterminizeLatticePrunedOptions).
///
/// This is kaldi's @kfile{src/lat/determinize-lattice-pruned.h} options
/// struct (the determinization algorithms themselves come from kaldifst).
///
/// @par Used by:
/// Callers that plug kaldifst's determinization into the raw lattices
/// produced by LatticeFasterDecoderTpl::GetRawLattice().
///
/// @see DeterminizeLatticePhonePrunedOptions
struct DeterminizeLatticePrunedOptions {
  /// A small offset used to measure equality of weights
  /// (kaldi @c delta).
  float delta;
  /// If > 0, determinization fails (returns false) when the algorithm's
  /// approximate memory consumption crosses this threshold
  /// (kaldi @c max_mem).
  int max_mem;
  /// If > 0, detects non-determinizable input -- a case that would not be
  /// caught by max_mem (kaldi @c max_loop).
  int max_loop;
  /// If > 0, fail when the number of states crosses this limit
  /// (kaldi @c max_states).
  int max_states;
  /// If > 0, fail when the number of arcs crosses this limit
  /// (kaldi @c max_arcs).
  int max_arcs;
  /// Retry cutoff for the pruned determinization (kaldi @c retry_cutoff).
  float retry_cutoff;
  explicit DeterminizeLatticePrunedOptions(float delta = fst::kDelta,
                                           int max_mem = -1, int max_loop = -1,
                                           int max_states = -1,
                                           int max_arcs = -1,
                                           float retry_cutoff = 0.5)
      : delta(delta),
        max_mem(max_mem),
        max_loop(max_loop),
        max_states(max_states),
        max_arcs(max_arcs),
        retry_cutoff(retry_cutoff) {}

  /// @return A human-readable "name(k=v, ...)" dump of the options.
  std::string ToString() const {
    std::ostringstream os;

    os << "DeterminizeLatticePrunedOptions(";
    os << "delta=" << delta << ", ";
    os << "max_mem=" << max_mem << ", ";
    os << "max_loop=" << max_loop << ", ";
    os << "max_states=" << max_states << ", ";
    os << "max_arcs=" << max_arcs << ", ";
    os << "retry_cutoff=" << retry_cutoff << ")";

    return os.str();
  }
};

/// @brief Options for phone-aware lattice determinization (kaldi's
/// DeterminizeLatticePhonePrunedOptions).
///
/// This is kaldi's @kfile{src/lat/determinize-lattice-pruned.h} options
/// struct used by DeterminizeLatticePhonePrunedWrapper in kaldi.
///
/// @par Used by:
/// Callers that determinize the raw lattices from
/// LatticeFasterDecoderTpl::GetRawLattice() phone-aware (kaldi's
/// compact-lattice output path).
///
/// @see DeterminizeLatticePrunedOptions
struct DeterminizeLatticePhonePrunedOptions {
  /// A small offset used to measure equality of weights; tolerance used in
  /// determinization (kaldi @c delta).
  float delta;
  /// If > 0, determinization fails (returns false) when the algorithm's
  /// approximate memory consumption crosses this threshold
  /// (kaldi @c max_mem).
  int max_mem;
  /// If true, do a first-pass determinization on both phones and words
  /// (kaldi @c phone_determinize).
  bool phone_determinize;
  /// If true, do a second-pass determinization on words only
  /// (kaldi @c word_determinize).
  bool word_determinize;
  /// If true, push weights and minimize after determinization
  /// (kaldi @c minimize).
  bool minimize;

  // Used as a default argument (= T()); explicit would break the
  // copy-initialization there.
  // NOLINTNEXTLINE -- implicit ctor on purpose; see comment above
  DeterminizeLatticePhonePrunedOptions(float delta = fst::kDelta,
                                       int32_t max_mem = 50000000,
                                       bool phone_determinize = true,
                                       bool word_determinize = true,
                                       bool minimize = false)
      : delta(delta),
        max_mem(max_mem),
        phone_determinize(phone_determinize),
        word_determinize(word_determinize),
        minimize(minimize) {}

  /// @return A human-readable "name(k=v, ...)" dump of the options.
  std::string ToString() const {
    std::ostringstream os;

    os << "DeterminizeLatticePhonePrunedOptions(";
    os << "delta=" << delta << ", ";

    os << "max_mem=" << max_mem << ", ";

    os << "phone_determinize=" << (phone_determinize ? "True" : "False")
       << ", ";

    os << "word_determinize=" << (word_determinize ? "True" : "False") << ", ";

    os << "minimize=" << (minimize ? "True" : "False") << ")";

    return os.str();
  }
};

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_DETERMINIZE_LATTICE_PRUNED_H_
