// kaldi-decoder/csrc/kaldi-math.cc

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Yanmin Qian;
//                      Jan Silovsky;  Saarland University
//                2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/base/kaldi-math.cc

#include "kaldi-decoder/csrc/kaldi-math.h"

#include <cstdlib>
#include <mutex>  // NOLINT

namespace kaldi_decoder {

RandomState::RandomState() {
  // we initialize it as Rand() + 27437 instead of just Rand(), because on some
  // systems, e.g. at the very least Mac OSX Yosemite and later, it seems to be
  // the case that rand_r when initialized with rand() will give you the exact
  // same sequence of numbers that rand() will give if you keep calling rand()
  // after that initial call.  This can cause problems with repeated sequences.
  // For example if you initialize two RandomState structs one after the other
  // without calling rand() in between, they would give you the same sequence
  // offset by one (if we didn't have the "+ 27437" in the code).  27437 is just
  // a randomly chosen prime number.
  seed = unsigned(Rand()) + 27437;
}

int32_t Rand(struct RandomState *state) {
#if !defined(_POSIX_THREAD_SAFE_FUNCTIONS)
  // On Windows and Cygwin, just call Rand()
  return rand();
#else
  if (state) {
    return rand_r(&(state->seed));
  } else {
    static std::mutex _RandMutex;
    std::lock_guard<std::mutex> lock(_RandMutex);
    return rand();
  }
#endif
}

}  // namespace kaldi_decoder
