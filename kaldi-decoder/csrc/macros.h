// kaldi-decoder/csrc/macros.h
//
// Copyright (c)  2026  Xiaomi Corporation

// This file is copied/modified from
// kaldi-native-fbank/kaldi-native-fbank/csrc/macros.h

#ifndef KALDI_DECODER_CSRC_MACROS_H_
#define KALDI_DECODER_CSRC_MACROS_H_

#include <cstdint>
#include <cstdio>

#if __ANDROID_API__ >= 8
#include "android/log.h"
#define KALDI_DECODER_LOGE(...)                                              \
  do {                                                                       \
    std::fprintf(stderr, "%s:%s:%d ", __FILE__, __func__,                    \
                 static_cast<int32_t>(__LINE__));                            \
    std::fprintf(stderr, __VA_ARGS__);                                       \
    std::fprintf(stderr, "\n");                                              \
    __android_log_print(ANDROID_LOG_WARN, "kaldi-decoder", "%s:%s:%d",       \
                        __FILE__, __func__, static_cast<int32_t>(__LINE__)); \
    __android_log_print(ANDROID_LOG_WARN, "kaldi-decoder", __VA_ARGS__);     \
  } while (0)
#elif defined(__OHOS__)
#include "hilog/log.h"
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x6666
#define LOG_TAG "kaldi-decoder"
#define KALDI_DECODER_LOGE(...) OH_LOG_INFO(LOG_APP, __VA_ARGS__)
#else
/**
 * printf-style logging of errors/warnings: prints ``file:func:line`` and the
 * message to stderr.  Usage: ``KALDI_DECODER_LOGE("bad value: %d", x);``
 */
#define KALDI_DECODER_LOGE(...)                           \
  do {                                                    \
    std::fprintf(stderr, "%s:%s:%d ", __FILE__, __func__, \
                 static_cast<int32_t>(__LINE__));         \
    std::fprintf(stderr, __VA_ARGS__);                    \
    std::fprintf(stderr, "\n");                           \
  } while (0)
#endif

#endif  // KALDI_DECODER_CSRC_MACROS_H_
