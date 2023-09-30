// kaldi-decoder/csrc/hash-list-test.cc

// Copyright 2009-2011     Microsoft Corporation
//                2013     Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023     Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/utils/hash-list-test.cc

#include "kaldi-decoder/csrc/hash-list.h"

#include <stdlib.h>

#include <map>  // for baseline.

// #include "kaldi-decoder/csrc/kaldi-math.h"
#include "gtest/gtest.h"

namespace kaldi_decoder {

template <class Int, class T>
void TestHashList() {
  typedef typename HashList<Int, T>::Elem Elem;

  HashList<Int, T> hash;
  hash.SetSize(200);  // must be called before use.
  std::map<Int, T> m1;

  for (size_t j = 0; j < 50; j++) {
    Int key = rand() % 200;  // NOLINT
    T val = rand() % 50;     // NOLINT
    m1[key] = val;
    Elem *e = hash.Find(key);
    if (e) {
      e->val = val;
    } else {
      hash.Insert(key, val);
    }
  }

  std::map<Int, T> m2;

  for (int i = 0; i < 100; i++) {
    m2.clear();
    for (auto iter = m1.begin(); iter != m1.end(); iter++) {
      m2[iter->first + 1] = iter->second;
    }
    std::swap(m1, m2);

    Elem *h = hash.Clear(), *tmp;

    // note, SetSize is relatively cheap
    hash.SetSize(100 + rand() % 100);  // NOLINT
    // operation as long as we are not increasing the size more than it's ever
    // previously been increased to.

    for (; h != nullptr; h = tmp) {
      hash.Insert(h->key + 1, h->val);
      tmp = h->tail;
      hash.Delete(h);  // think of this like calling delete.
    }

    // Now make sure h and m2 are the same.
    const Elem *list = hash.GetList();
    size_t count = 0;
    for (; list != nullptr; list = list->tail, count++) {
      KALDI_DECODER_ASSERT(m1[list->key] == list->val);
    }

    for (size_t j = 0; j < 10; j++) {
      Int key = rand() % 200;  // NOLINT
      bool found_m1 = (m1.find(key) != m1.end());

      if (found_m1) m1[key];

      Elem *e = hash.Find(key);
      KALDI_DECODER_ASSERT((e != nullptr) == found_m1);

      if (found_m1) KALDI_DECODER_ASSERT(m1[key] == e->val);
    }

    KALDI_DECODER_ASSERT(m1.size() == count);
  }

  Elem *h = hash.Clear(), *tmp;
  for (; h != nullptr; h = tmp) {
    tmp = h->tail;
    hash.Delete(h);  // think of this like calling delete.
  }
}

TEST(HashList, Test) {
  for (size_t i = 0; i < 3; i++) {
    TestHashList<int, unsigned int>();
    TestHashList<unsigned int, int>();
    TestHashList<int16_t, int32_t>();
    TestHashList<int16_t, int32_t>();
    TestHashList<char, unsigned char>();
    TestHashList<unsigned char, int>();
  }
}

}  // namespace kaldi_decoder
