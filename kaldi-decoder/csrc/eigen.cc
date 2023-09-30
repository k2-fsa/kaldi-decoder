// kaldi-decoder/csrc/eigen.cc

// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-decoder/csrc/eigen.h"

#include <cmath>
#include <random>

namespace kaldi_decoder {

// see https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
// log(sum(exp(x))) == log(sum(exp(x - max(x)))) + max(x)
float LogSumExp(const FloatVector &v) {
  float max_v = v.maxCoeff();

  return std::log((v.array() - max_v).exp().sum()) + max_v;
}

FloatVector Softmax(const FloatVector &v, float *log_sum_exp /*= nullptr*/) {
  float max_v = v.maxCoeff();

  FloatVector ans = (v.array() - max_v).exp();

  float ans_sum = ans.sum();

  if (log_sum_exp) {
    *log_sum_exp = std::log(ans_sum) + max_v;
  }

  return ans / ans_sum;
}

FloatVector RandnVector(int32_t n, float mean /*= 0*/, float stddev /*= 1*/) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> d{mean, stddev};

  FloatVector ans(n);

  for (int32_t i = 0; i != n; ++i) {
    ans[i] = d(gen);
  }

  return ans;
}

FloatMatrix RandnMatrix(int32_t rows, int32_t cols, float mean /*= 0*/,
                        float stddev /*= 1*/) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> d{mean, stddev};

  FloatMatrix ans(rows, cols);

  for (int32_t i = 0; i != ans.size(); ++i) {
    ans(i) = d(gen);
  }

  return ans;
}

float Randn(float mean /*= 0*/, float stddev /*= 1*/) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> d{mean, stddev};

  return d(gen);
}

}  // namespace kaldi_decoder
