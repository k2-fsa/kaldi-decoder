add_subdirectory(csrc)

if(KALDI_DECODER_BUILD_PYTHON)
  add_subdirectory(python)
endif()
