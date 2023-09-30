// kaldi-decoder/csrc/eigen-test.cc

// Copyright (c)  2023     Xiaomi Corporation

#include "kaldi-decoder/csrc/eigen.h"

#include <cmath>
#include <type_traits>

#include "gtest/gtest.h"

// See
//
// Quick reference guide
// https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
//
// Preprocessor directives
// https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
//
// Understanding Eigen
// https://eigen.tuxfamily.org/dox/UserManual_UnderstandingEigen.html
//
// Using Eigen in CMake Projects
// https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html

namespace kaldi_decoder {

TEST(Eigen, Hello) {
  Eigen::MatrixXd m(2, 2);  // uninitialized; contains garbage data
  EXPECT_EQ(m.size(), 2 * 2);
  EXPECT_EQ(m.rows(), 2);
  EXPECT_EQ(m.cols(), 2);

  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  auto m2 = m;  // value semantics; create a copy
  m2(0, 0) = 10;
  EXPECT_EQ(m(0, 0), 3);

  Eigen::MatrixXd m3 = std::move(m2);
  // now m2 is empty
  EXPECT_EQ(m2.size(), 0);
  EXPECT_EQ(m3(0, 0), 10);

  double *d = &m(0, 0);
  d[0] = 11;
  d[1] = 20;
  d[2] = 30;
  d[3] = 40;
  EXPECT_EQ(m(0, 0), 11);  // column major by default
  EXPECT_EQ(m(1, 0), 20);
  EXPECT_EQ(m(0, 1), 30);  // it is contiguous in memory
  EXPECT_EQ(m(1, 1), 40);

  // column major
  EXPECT_EQ(m(0), 11);
  EXPECT_EQ(m(1), 20);
  EXPECT_EQ(m(2), 30);
  EXPECT_EQ(m(3), 40);

  Eigen::MatrixXf a;
  EXPECT_EQ(a.size(), 0);

  Eigen::Matrix3f b;  // uninitialized
  EXPECT_EQ(b.size(), 3 * 3);

  Eigen::MatrixXf c(2, 5);  // uninitialized
  EXPECT_EQ(c.size(), 2 * 5);

  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 5);

  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f{
        {1, 2},
        {3, 4},
    };

    // row major
    EXPECT_EQ(f(0), 1);
    EXPECT_EQ(f(1), 2);
    EXPECT_EQ(f(2), 3);
    EXPECT_EQ(f(3), 4);

    // Note: f[0] causes compilation errors
  }
}

TEST(Eigen, Identity) {
  auto m = Eigen::Matrix3f::Identity();  // 3x3 identity matrix
  EXPECT_EQ(m.sum(), 3);

  auto n = Eigen::MatrixXf::Identity(2, 3);  // 2x3 identity matrix
#if 0
  1 0 0
  0 1 0
#endif
}

// https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae814abb451b48ed872819192dc188c19
TEST(Eigen, Random) {
  // Random: Uniform distribution in the range [-1, 1]
  auto m = Eigen::MatrixXd::Random(2, 3);
#if 0
  -0.999984   0.511211  0.0655345
  -0.736924 -0.0826997  -0.562082
#endif

  // Note: We don't need to specify the shape for Random() in this case
  auto m2 = Eigen::Matrix3d::Random();
#if 0
    -0.999984 -0.0826997  -0.905911
    -0.736924  0.0655345   0.357729
    0.511211  -0.562082   0.358593
#endif
}

TEST(Eigen, Vector) {
  Eigen::VectorXd v(3);

  // comma initializer
  v << 1, 2, 3;
  EXPECT_EQ(v(0), 1);
  EXPECT_EQ(v(1), 2);
  EXPECT_EQ(v(2), 3);

  // vector also support operator[]
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);

  double *p = &v[0];
  p[0] = 10;
  p[1] = 20;
  p[2] = 30;

  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);

  // fixed size
  Eigen::Vector3d a(10, 20, 30);
  EXPECT_EQ(a[0], 10);
  EXPECT_EQ(a[1], 20);
  EXPECT_EQ(a[2], 30);
}

TEST(Eigen, CommaInitializer) {
  // comma initializer does not depend on the storage major
  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m(2,
                                                                            2);
    m << 1, 2, 3, 4;
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(1, 0), 3);
    EXPECT_EQ(m(1, 1), 4);
  }

  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(2,
                                                                            2);
    m << 1, 2, 3, 4;
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(1, 0), 3);
    EXPECT_EQ(m(1, 1), 4);
  }
}

TEST(Eigen, Resize) {
  // a resize operation is a destructive operation if it changes the size.
  // The original content is not copied to the resized area
  Eigen::MatrixXf a(2, 3);
  EXPECT_EQ(a.rows(), 2);
  EXPECT_EQ(a.cols(), 3);
  EXPECT_EQ(a.size(), a.rows() * a.cols());

  a.resize(5, 6);
  EXPECT_EQ(a.rows(), 5);
  EXPECT_EQ(a.cols(), 6);
  EXPECT_EQ(a.size(), a.rows() * a.cols());

  Eigen::MatrixXf b;
  EXPECT_EQ(b.size(), 0);

  b = a;  // copy by value
  EXPECT_EQ(b.rows(), 5);
  EXPECT_EQ(b.cols(), 6);
}

TEST(Eigen, MatMul) {
  Eigen::MatrixXf a(2, 2);
  a << 1, 2, 3, 4;

  Eigen::MatrixXf b(2, 2);
  b << 3, 0, 0, 2;

  Eigen::MatrixXf c = a * b;  // matrix multiplication
  EXPECT_EQ(c(0, 0), a(0, 0) * b(0, 0));
  EXPECT_EQ(c(0, 1), a(0, 1) * b(1, 1));

  EXPECT_EQ(c(1, 0), a(1, 0) * b(0, 0));
  EXPECT_EQ(c(1, 1), a(1, 1) * b(1, 1));

  Eigen::MatrixXf d;
  d.noalias() = a * b;  // explicitly specify that there is no alias

  EXPECT_EQ(d(0, 0), a(0, 0) * b(0, 0));
  EXPECT_EQ(d(0, 1), a(0, 1) * b(1, 1));

  EXPECT_EQ(d(1, 0), a(1, 0) * b(0, 0));
  EXPECT_EQ(d(1, 1), a(1, 1) * b(1, 1));
}

TEST(Eigen, Transpose) {
  Eigen::MatrixXf a(2, 2);
  a << 1, 2, 3, 4;

  // a = a.transpose();  // wrong due to alias
#if 0
  1 2
  2 4
#endif

  Eigen::MatrixXf b(2, 2);
  b << 1, 2, 3, 4;
  b.transposeInPlace();  // correct
#if 0
  1 3
  2 4
#endif
}

TEST(Eigen, Reduction) {
  Eigen::MatrixXf m(2, 2);
  m << 1, 2, 3, -5;
#if 0
  1 2
  3 -5
#endif

  EXPECT_EQ(m.sum(), 1);
  EXPECT_EQ(m.prod(), -30);
  EXPECT_EQ(m.mean(), m.sum() / m.size());
  EXPECT_EQ(m.minCoeff(), -5);
  EXPECT_EQ(m.maxCoeff(), 3);
  EXPECT_EQ(m.trace(), 1 + (-5));
  EXPECT_EQ(m.trace(), m.diagonal().sum());

  std::ptrdiff_t row_id, col_id;

  float a = m.minCoeff(&row_id, &col_id);
  EXPECT_EQ(a, -5);
  EXPECT_EQ(row_id, 1);
  EXPECT_EQ(col_id, 1);

  float b = m.maxCoeff(&row_id, &col_id);
  EXPECT_EQ(b, 3);
  EXPECT_EQ(row_id, 1);
  EXPECT_EQ(col_id, 0);
}

TEST(Eigen, Array) {
  // Note: It is XX for a 2-D array
  Eigen::ArrayXXf a(2, 3);
  a << 1, 2, 3, 4, 5, 6;
#if 0
  1 2 3
  4 5 6
#endif

  EXPECT_EQ(a(0, 0), 1);
  EXPECT_EQ(a(0, 1), 2);
  EXPECT_EQ(a(0, 2), 3);

  EXPECT_EQ(a(1, 0), 4);
  EXPECT_EQ(a(1, 1), 5);
  EXPECT_EQ(a(1, 2), 6);

  EXPECT_EQ(a.rows(), 2);
  EXPECT_EQ(a.cols(), 3);

  Eigen::Array<float, 5, 2> b;
  EXPECT_EQ(b.rows(), 5);
  EXPECT_EQ(b.cols(), 2);

  // 1-d array
  Eigen::ArrayXf c(10);

  EXPECT_EQ(c.rows(), 10);
  EXPECT_EQ(c.cols(), 1);
  EXPECT_EQ(c.size(), 10);
  static_assert(
      std::is_same<decltype(c), Eigen::Array<float, Eigen::Dynamic, 1>>::value,
      "");

  static_assert(std::is_same<Eigen::ArrayXf,
                             Eigen::Array<float, Eigen::Dynamic, 1>>::value,
                "");

  static_assert(
      std::is_same<Eigen::ArrayXXf,
                   Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>::value,
      "");

  static_assert(std::is_same<Eigen::Array3f, Eigen::Array<float, 3, 1>>::value,
                "");

  static_assert(std::is_same<Eigen::Array33f, Eigen::Array<float, 3, 3>>::value,
                "");
}

TEST(Eigen, ArrayMultiplication) {
  Eigen::ArrayXXf a(2, 2);
  a << 1, 2, 3, 4;
  Eigen::ArrayXXf b = a * a;

  EXPECT_EQ(b(0, 0), a(0, 0) * a(0, 0));
  EXPECT_EQ(b(0, 1), a(0, 1) * a(0, 1));
  EXPECT_EQ(b(1, 0), a(1, 0) * a(1, 0));
  EXPECT_EQ(b(1, 1), a(1, 1) * a(1, 1));

  // column-wise product
  Eigen::ArrayXXf c = a.matrix().cwiseProduct(a.matrix());

  EXPECT_EQ(c(0, 0), a(0, 0) * a(0, 0));
  EXPECT_EQ(c(0, 1), a(0, 1) * a(0, 1));
  EXPECT_EQ(c(1, 0), a(1, 0) * a(1, 0));
  EXPECT_EQ(c(1, 1), a(1, 1) * a(1, 1));
}

TEST(Eigen, CoefficientWise) {
  Eigen::ArrayXXf a(2, 2);
  a << 1, 2, 3, -4;

  EXPECT_EQ(a.abs()(1, 1), 4);
  EXPECT_EQ(a.abs().sum(), 10);

  EXPECT_EQ(a.abs().sqrt()(1, 1), 2);
}

TEST(Eigen, Row) {
  Eigen::MatrixXf m(2, 3);
  m << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXf a = m.row(0);  // copied to a
  EXPECT_EQ(a.rows(), 1);
  EXPECT_EQ(a.cols(), 3);

  a(0) = 10;
  EXPECT_EQ(m(0, 0), 1);

  Eigen::MatrixXf b = m.col(1);  // copied to b
  EXPECT_EQ(b.rows(), 2);
  EXPECT_EQ(b.cols(), 1);
  b(0) = 10;
  EXPECT_EQ(m(0, 1), 2);

  auto c = m.row(0);  // c is a proxy object; no copy is created
  c(0) = 10;          // also change m
  EXPECT_EQ(m(0, 0), 10);
  EXPECT_EQ(c.rows(), 1);
  EXPECT_EQ(c.cols(), 3);

  auto d = c;  // d is also a proxy
  d(0) = 100;
  EXPECT_EQ(c(0), 100);
  EXPECT_EQ(m(0), 100);

  // N5Eigen6MatrixIfLin1ELin1ELi0ELin1ELin1EEE
  // std::cout << typeid(m).name() << "\n";

  // N5Eigen6MatrixIfLin1ELin1ELi0ELin1ELin1EEE
  // std::cout << typeid(b).name() << "\n";

  // N5Eigen5BlockINS_6MatrixIfLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEE
  // std::cout << typeid(c).name() << "\n";
}

TEST(Eigen, Sequence) {
  // (start, end)
  // Note that 5 is included here
  auto seq = Eigen::seq(2, 5);  // [2, 3, 4, 5]
  EXPECT_EQ(seq.size(), 4);
  for (int32_t i = 0; i != seq.size(); ++i) {
    EXPECT_EQ(seq[i], i + 2);
  }

  // start 2, end 5, increment 2,
  // (start, end, increment), note that 5 is not included here
  auto seq2 = Eigen::seq(2, 5, 2);  // [2, 4]
  EXPECT_EQ(seq2.size(), 2);
  EXPECT_EQ(seq2[0], 2);
  EXPECT_EQ(seq2[1], 4);

  // (start, sequence_length)
  auto seq3 = Eigen::seqN(2, 5);  // [2, 3, 4, 5, 6]
  EXPECT_EQ(seq3.size(), 5);
  for (int32_t i = 0; i != seq3.size(); ++i) {
    EXPECT_EQ(seq3[i], i + 2);
  }

  Eigen::VectorXf v(5);
  v << 0, 1, 2, 3, 4;

  Eigen::VectorXf a = v(Eigen::seq(2, Eigen::last));
  EXPECT_EQ(a.size(), 3);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);

  a = v(Eigen::seq(2, Eigen::last - 1));
  EXPECT_EQ(a.size(), 2);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
}

TEST(Eigen, CopyRow) {
  Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 3);
  Eigen::MatrixXf b(2, 3);
  b.row(0) = a.row(0);
  b.row(1) = a.row(1);
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a(i), b(i));
  }

  a = Eigen::MatrixXf::Random(5, 3);
  b.resize(5, 3);

  b(Eigen::seqN(0, 3), Eigen::all) = a(Eigen::seqN(0, 3), Eigen::all);
  b(Eigen::seqN(3, 2), Eigen::all) = a(Eigen::seqN(3, 2), Eigen::all);
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a(i), b(i));
  }

  Eigen::MatrixXf c(5, 3);
  c(Eigen::seqN(0, 5), Eigen::all) = a;
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a(i), c(i));
  }
}

TEST(Eigen, SpecialFunctions) {
  Eigen::MatrixXf a(2, 3);
  a.setOnes();
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a(i), 1);
  }

  a.setZero();
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a(i), 0);
  }
}

TEST(Eigen, LogSumExp) {
  Eigen::VectorXf v(5);
  v << 0.1, 0.3, 0.2, 0.15, 0.25;
  auto f = LogSumExp(v);
  EXPECT_NEAR(f, 1.8119, 1e-4);

  v.resize(10);
  v << -0.028933119028806686, -0.8265501260757446, 0.31104734539985657,
      0.25977903604507446, 0.18070533871650696, 0.02222185768187046,
      -1.4124598503112793, -0.5896500945091248, -0.17299121618270874,
      -0.6516317129135132;

  f = LogSumExp(v);
  EXPECT_NEAR(f, 2.1343, 1e-4);
}

TEST(Eigen, Addmm) {
  // means_invvars_: (nmix, dim)
  // data: (dim,)
  // loglikes: (nmix,)
  // loglikes +=  means * inv(vars) * data.
  // loglikes->AddMatVec(1.0, means_invvars_, kNoTrans, data, 1.0);
  // loglikes = loglikes.unsqueeze(1);  // (nmix, 1)
  // loglikes.addmm_(means_invvars_, data.unsqueeze(1));

  int32_t nmix = 3;
  int32_t dim = 5;

  Eigen::MatrixXf means_invvars = Eigen::MatrixXf::Random(nmix, dim);
  Eigen::VectorXf data = Eigen::VectorXf::Random(dim);

  Eigen::VectorXf loglikes = Eigen::VectorXf::Random(nmix);

  loglikes += means_invvars * data;
}

TEST(Eigen, VectorOp) {
  Eigen::VectorXf a(2);
  a << 1, 2;
  Eigen::RowVectorXf b(2);
  b << 10, 20;

  {
    Eigen::VectorXf c = a.array() + b.transpose().array();
    EXPECT_EQ(c.size(), 2);
    EXPECT_EQ(c[0], a[0] + b[0]);
    EXPECT_EQ(c[1], a[1] + b[1]);
  }

#if 0
  {
    // Don't do this!
    Eigen::VectorXf c = a.array() + b.array();
    EXPECT_EQ(c.size(), 1);
    EXPECT_EQ(c[0], a[0] + b[0]);
  }

  {
    // Don't do this!
    Eigen::RowVectorXf c(2);
    c << 100, 200;
    c.row(0) = a.array() + b.array();
    EXPECT_EQ(c.size(), 2);
    EXPECT_EQ(c[0], a[0] + b[0]);
    EXPECT_EQ(c[1], a[1] + b[0]);
  }

  {
    // Don't do this!
    Eigen::RowVectorXf c(2);
    c << 100, 200;
    c.row(0) = b.array() + a.array();
    EXPECT_EQ(c.size(), 2);
    EXPECT_EQ(c[0], b[0] + a[0]);
    EXPECT_EQ(c[1], b[1] + a[0]);
  }
#endif
}

TEST(Eigen, VectorOp2) {
  Eigen::MatrixXf m(2, 3);
  m << 1, 4, 8, 16, 9, 25;

  Eigen::VectorXf v(3);
  v << 10, 20, 30;

  v = v.transpose().array() * m.row(1).array().sqrt();
  EXPECT_EQ(v.size(), 3);

  EXPECT_EQ(v[0], 10 * std::sqrt(16));
  EXPECT_EQ(v[1], 20 * std::sqrt(9));
  EXPECT_EQ(v[2], 30 * std::sqrt(25));
}

TEST(Eigen, RowwiseSum) {
  Eigen::MatrixXf m(2, 3);
  m << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXf a = m.rowwise().sum();
  EXPECT_EQ(a.rows(), m.rows());
  EXPECT_EQ(a.cols(), 1);

  EXPECT_EQ(a(0), 1 + 2 + 3);
  EXPECT_EQ(a(1), 4 + 5 + 6);

  Eigen::MatrixXf b = m.colwise().sum();
  EXPECT_EQ(b.rows(), 1);
  EXPECT_EQ(b.cols(), m.cols());

  EXPECT_EQ(b(0), 1 + 4);
  EXPECT_EQ(b(1), 2 + 5);
  EXPECT_EQ(b(2), 3 + 6);

  // assign a row vector to a vector
  Eigen::VectorXf c = m.colwise().sum();
  EXPECT_EQ(c.rows(), b.cols());
  EXPECT_EQ(c.cols(), 1);

  EXPECT_EQ(c(0), 1 + 4);
  EXPECT_EQ(c(1), 2 + 5);
  EXPECT_EQ(c(2), 3 + 6);

  // assign a vector to a row vector
  Eigen::RowVectorXf d = m.rowwise().sum();
  EXPECT_EQ(d(0), 1 + 2 + 3);
  EXPECT_EQ(d(1), 4 + 5 + 6);

  // now for array
  Eigen::MatrixXf a2 = m.array().rowwise().sum();
  EXPECT_EQ(a2.rows(), m.rows());
  EXPECT_EQ(a2.cols(), 1);
}

TEST(Eigen, Replicate) {
  Eigen::VectorXf v(2);
  v << 1, 2;
  Eigen::VectorXf a = v.replicate(3, 1);
  EXPECT_EQ(a.size(), v.size() * 3);
  EXPECT_EQ(a[0], v[0]);
  EXPECT_EQ(a[1], v[1]);
  EXPECT_EQ(a[2], v[0]);
  EXPECT_EQ(a[3], v[1]);
  EXPECT_EQ(a[4], v[0]);
  EXPECT_EQ(a[5], v[1]);

  Eigen::MatrixXf m = v.transpose().replicate(3, 1);
  // repeat the rows 3 times
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), v.size());

  Eigen::MatrixXf expected_m(3, 2);
  expected_m << 1, 2, 1, 2, 1, 2;
  for (int32_t i = 0; i != m.size(); ++i) {
    EXPECT_EQ(m(i), expected_m(i));
  }
}

TEST(Eigen, Indexes) {
  Eigen::VectorXf v(5);
  v << 0, 10, 20, 30, 40;

  std::vector<int32_t> indexes = {1, 4, 0, 2, 1};
  Eigen::VectorXf a = v(indexes);
  EXPECT_EQ(a.size(), indexes.size());
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a[i], v[indexes[i]]);
  }

  Eigen::MatrixXf m(3, 2);
  m << 0, 1, 2, 3, 4, 5;

  indexes = {1, 0, 2, 1};
  Eigen::MatrixXf b = m(indexes, Eigen::all);
#if 0
    2 3
    0 1
    4 5
    2 3
#endif
}

TEST(Eigen, TestSoftmax) {
  Eigen::VectorXf v(5);
  v << 0.46589261293411255, 0.5329158902168274, 0.45468050241470337,
      0.509181022644043, 0.4529399275779724;

  Eigen::VectorXf expected(5);
  expected << 0.1964813768863678, 0.21010152995586395, 0.19429071247577667,
      0.205173522233963, 0.19395282864570618;

  Eigen::VectorXf actual = Softmax(v);
  for (int32_t i = 0; i != 5; ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-4);
  }
}

TEST(Eigen, Op1) {
  Eigen::VectorXf a(2);
  Eigen::VectorXf b(3);

  a << 10, 20;
  b << 3, 5, 8;

  Eigen::MatrixXf c;
  c = a * b.transpose();

  Eigen::MatrixXf expected(2, 3);
  expected << 30, 50, 80, 60, 100, 160;
  for (int32_t i = 0; i != 6; ++i) {
    EXPECT_EQ(c(i), expected(i));
  }
}

TEST(Eigen, Op2) {
  Eigen::MatrixXf a(2, 3);
  a << 1, 2, 3, 4, 5, 6;

  Eigen::VectorXf b(2);
  b << 10, 20;

  a = a.array() * b.replicate(1, a.cols()).array();

  Eigen::MatrixXf expected(2, 3);
  expected << 10, 20, 30, 80, 100, 120;
  for (int32_t i = 0; i != 6; ++i) {
    EXPECT_EQ(a(i), expected(i));
  }

  std::cout << a << "\n";
}

TEST(Eigen, Op3) {
  Eigen::MatrixXf a(2, 3);
  a << 1, 2, 3, 4, 5, 6;
  Eigen::VectorXf b = a.row(1);
  // b contains [4, 5, 6] and is a column vector
  b[0] = 100;

  // it is OK to assign a column vector to a row vector
  a.row(1) = b;
  // a is
  // 1 2 3
  // 100 5 6
}

TEST(Eigen, Dot) {
  Eigen::VectorXf a(3);
  Eigen::VectorXf b(3);
  a << 1, 2, 3;
  b << 4, 5, 6;
  float c = a.dot(b);
  EXPECT_EQ(c, 1 * 4 + 2 * 5 + 3 * 6);
}

}  // namespace kaldi_decoder
