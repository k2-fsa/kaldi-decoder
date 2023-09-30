// kaldi-decoder/csrc/log.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef KALDI_DECODER_CSRC_LOG_H_
#define KALDI_DECODER_CSRC_LOG_H_

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace kaldi_decoder {

enum class LogLevel {
  kInfo = 0,
  kWarn = 1,
  kError = 2,  // abort the program
};

class Logger {
 public:
  Logger(const char *filename, const char *func_name, uint32_t line_num,
         LogLevel level)
      : level_(level) {
    os_ << filename << ":" << func_name << ":" << line_num << "\n";
    switch (level_) {
      case LogLevel::kInfo:
        os_ << "[I] ";
        break;
      case LogLevel::kWarn:
        os_ << "[W] ";
        break;
      case LogLevel::kError:
        os_ << "[E] ";
        break;
    }
  }

  template <typename T>
  Logger &operator<<(const T &val) {
    os_ << val;
    return *this;
  }

  ~Logger() noexcept(false) {
    if (level_ == LogLevel::kError) {
      // throw std::runtime_error(os_.str());
      // abort();
      throw std::runtime_error(os_.str());
    }
    // fprintf(stderr, "%s\n", os_.str().c_str());
  }

 private:
  std::ostringstream os_;
  LogLevel level_;
};

class Voidifier {
 public:
  void operator&(const Logger &) const {}
};

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__PRETTY_FUNCTION__)
// for clang and GCC
#define KALDI_DECODER_FUNC __PRETTY_FUNCTION__
#else
// for other compilers
#define KALDI_DECODER_FUNC __func__
#endif

#define KALDI_DECODER_LOG                                       \
  kaldi_decoder::Logger(__FILE__, KALDI_DECODER_FUNC, __LINE__, \
                        kaldi_decoder::LogLevel::kInfo)

#define KALDI_DECODER_WARN                                      \
  kaldi_decoder::Logger(__FILE__, KALDI_DECODER_FUNC, __LINE__, \
                        kaldi_decoder::LogLevel::kWarn)

#define KALDI_DECODER_ERR                                       \
  kaldi_decoder::Logger(__FILE__, KALDI_DECODER_FUNC, __LINE__, \
                        kaldi_decoder::LogLevel::kError)

#define KALDI_DECODER_ASSERT(x)                                             \
  (x) ? (void)0                                                             \
      : kaldi_decoder::Voidifier() & KALDI_DECODER_ERR << "Check failed!\n" \
                                                       << "x: " << #x

#define KALDI_DECODER_PARANOID_ASSERT KALDI_DECODER_ASSERT

#define KALDI_DECODER_DISALLOW_COPY_AND_ASSIGN(Class) \
 public:                                              \
  Class(const Class &) = delete;                      \
  Class &operator=(const Class &) = delete;

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_LOG_H_
