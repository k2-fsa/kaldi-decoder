add_subdirectory(csrc)

if(KALDI_DECODER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
