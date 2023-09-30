// kaldi-decoder/csrc/eigen.h

// Copyright (c)  2023  Xiaomi Corporation
#ifndef KALDI_DECODER_CSRC_EIGEN_H_
#define KALDI_DECODER_CSRC_EIGEN_H_

#include "Eigen/Dense"

namespace kaldi_decoder {

using FloatMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using DoubleMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using FloatVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using FloatRowVector = Eigen::Matrix<float, 1, Eigen::Dynamic>;

using DoubleRowVector = Eigen::Matrix<double, 1, Eigen::Dynamic>;

float LogSumExp(const FloatVector &v);

FloatVector Softmax(const FloatVector &v, float *log_sum_exp = nullptr);

// A vector of normal distribution
FloatVector RandnVector(int32_t n, float mean = 0, float stddev = 1);

FloatMatrix RandnMatrix(int32_t rows, int32_t cols, float mean = 0,
                        float stddev = 1);

float Randn(float mean = 0, float stddev = 1);

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_EIGEN_H_
