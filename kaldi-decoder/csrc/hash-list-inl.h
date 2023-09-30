// kaldi-decoder/csrc/hash-list-inl.h

// Copyright 2009-2011   Microsoft Corporation
//                2013   Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/utils/hash-list-inl.h

#ifndef KALDI_DECODER_CSRC_HASH_LIST_INL_H_
#define KALDI_DECODER_CSRC_HASH_LIST_INL_H_

// Do not include this file directly.  It is included by fast-hash.h

namespace kaldi_decoder {

template <class I, class T>
HashList<I, T>::HashList() {
  list_head_ = nullptr;
  bucket_list_tail_ = static_cast<size_t>(-1);  // invalid.
  hash_size_ = 0;
  freed_head_ = nullptr;
}

template <class I, class T>
void HashList<I, T>::SetSize(size_t size) {
  hash_size_ = size;
  KALDI_DECODER_ASSERT(list_head_ == nullptr &&
                       bucket_list_tail_ ==
                           static_cast<size_t>(-1));  // make sure empty.

  if (size > buckets_.size()) {
    buckets_.resize(size, HashBucket(0, nullptr));
  }
}

template <class I, class T>
typename HashList<I, T>::Elem *HashList<I, T>::Clear() {
  // Clears the hashtable and gives ownership of the currently contained list
  // to the user.
  for (size_t cur_bucket = bucket_list_tail_;
       cur_bucket != static_cast<size_t>(-1);
       cur_bucket = buckets_[cur_bucket].prev_bucket) {
    buckets_[cur_bucket].last_elem =
        nullptr;  // this is how we indicate "empty".
  }
  bucket_list_tail_ = static_cast<size_t>(-1);
  Elem *ans = list_head_;
  list_head_ = nullptr;
  return ans;
}

template <class I, class T>
const typename HashList<I, T>::Elem *HashList<I, T>::GetList() const {
  return list_head_;
}

template <class I, class T>
inline void HashList<I, T>::Delete(Elem *e) {
  e->tail = freed_head_;
  freed_head_ = e;
}

template <class I, class T>
inline typename HashList<I, T>::Elem *HashList<I, T>::Find(I key) {
  size_t index = (static_cast<size_t>(key) % hash_size_);
  HashBucket &bucket = buckets_[index];
  if (bucket.last_elem == nullptr) {
    return nullptr;  // empty bucket.
  } else {
    Elem *head = (bucket.prev_bucket == static_cast<size_t>(-1)
                      ? list_head_
                      : buckets_[bucket.prev_bucket].last_elem->tail),
         *tail = bucket.last_elem->tail;
    for (Elem *e = head; e != tail; e = e->tail) {
      if (e->key == key) {
        return e;
      }
    }

    return nullptr;  // Not found.
  }
}

template <class I, class T>
inline typename HashList<I, T>::Elem *HashList<I, T>::New() {
  if (freed_head_) {
    Elem *ans = freed_head_;
    freed_head_ = freed_head_->tail;
    return ans;
  } else {
    Elem *tmp = new Elem[allocate_block_size_];
    for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
      tmp[i].tail = tmp + i + 1;
    }

    tmp[allocate_block_size_ - 1].tail = nullptr;
    freed_head_ = tmp;
    allocated_.push_back(tmp);
    return this->New();
  }
}

template <class I, class T>
HashList<I, T>::~HashList() {
  // First test whether we had any memory leak within the
  // HashList, i.e. things for which the user did not call Delete().
  size_t num_in_list = 0, num_allocated = 0;

  for (Elem *e = freed_head_; e != nullptr; e = e->tail) {
    num_in_list++;
  }

  for (size_t i = 0; i < allocated_.size(); i++) {
    num_allocated += allocate_block_size_;
    delete[] allocated_[i];
  }

  if (num_in_list != num_allocated) {
    KALDI_DECODER_WARN << "Possible memory leak: " << num_in_list
                       << " != " << num_allocated
                       << ": you might have forgotten to call Delete on "
                       << "some Elems";
  }
}

template <class I, class T>
inline typename HashList<I, T>::Elem *HashList<I, T>::Insert(I key, T val) {
  size_t index = (static_cast<size_t>(key) % hash_size_);
  HashBucket &bucket = buckets_[index];
  // Check the element is existing or not.
  if (bucket.last_elem != nullptr) {
    Elem *head = (bucket.prev_bucket == static_cast<size_t>(-1)
                      ? list_head_
                      : buckets_[bucket.prev_bucket].last_elem->tail),
         *tail = bucket.last_elem->tail;

    for (Elem *e = head; e != tail; e = e->tail) {
      if (e->key == key) {
        return e;
      }
    }
  }

  // This is a new element. Insert it.
  Elem *elem = New();
  elem->key = key;
  elem->val = val;

  if (bucket.last_elem == nullptr) {  // Unoccupied bucket.  Insert at
    // head of bucket list (which is tail of regular list, they go in
    // opposite directions).
    if (bucket_list_tail_ == static_cast<size_t>(-1)) {
      // list was empty so this is the first elem.
      KALDI_DECODER_ASSERT(list_head_ == nullptr);
      list_head_ = elem;
    } else {
      // link in to the chain of Elems
      buckets_[bucket_list_tail_].last_elem->tail = elem;
    }
    elem->tail = nullptr;
    bucket.last_elem = elem;
    bucket.prev_bucket = bucket_list_tail_;
    bucket_list_tail_ = index;
  } else {
    // Already-occupied bucket.  Insert at tail of list of elements within
    // the bucket.
    elem->tail = bucket.last_elem->tail;
    bucket.last_elem->tail = elem;
    bucket.last_elem = elem;
  }
  return elem;
}

template <class I, class T>
void HashList<I, T>::InsertMore(I key, T val) {
  size_t index = (static_cast<size_t>(key) % hash_size_);
  HashBucket &bucket = buckets_[index];
  Elem *elem = New();
  elem->key = key;
  elem->val = val;

  // assume one element is already here
  KALDI_DECODER_ASSERT(bucket.last_elem != nullptr);

  if (bucket.last_elem->key == key) {  // standard behavior: add as last element
    elem->tail = bucket.last_elem->tail;
    bucket.last_elem->tail = elem;
    bucket.last_elem = elem;
    return;
  }
  Elem *e = (bucket.prev_bucket == static_cast<size_t>(-1)
                 ? list_head_
                 : buckets_[bucket.prev_bucket].last_elem->tail);
  // find place to insert in linked list
  while (e != bucket.last_elem->tail && e->key != key) {
    e = e->tail;
  }

  KALDI_DECODER_ASSERT(e->key == key);  // not found? - should not happen
  elem->tail = e->tail;
  e->tail = elem;
}

}  // namespace kaldi_decoder

#endif  // KALDI_DECODER_CSRC_HASH_LIST_INL_H_
