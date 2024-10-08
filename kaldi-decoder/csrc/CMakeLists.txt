include_directories(${CMAKE_SOURCE_DIR})

# Please keep the source files alphabetically sorted
set(srcs
  decodable-ctc.cc
  eigen.cc
  faster-decoder.cc
  lattice-faster-decoder.cc
  lattice-simple-decoder.cc
  simple-decoder.cc
)

# Always static build
add_library(kaldi-decoder-core STATIC ${srcs})

target_link_libraries(kaldi-decoder-core PUBLIC kaldifst_core)
target_link_libraries(kaldi-decoder-core PUBLIC Eigen3::Eigen)

if(KALDI_DECODER_ENABLE_TESTS)
  set(test_srcs
    eigen-test.cc
    hash-list-test.cc
  )

  function(kaldi_decoder_add_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name "${name}")
    add_executable(${target_name} ${source})
    target_link_libraries(${target_name}
      PRIVATE
      gtest
      gtest_main
      kaldi-decoder-core
    )
    add_test(NAME "Test.${target_name}"
      COMMAND
      $<TARGET_FILE:${target_name}>
    )
  endfunction()

  foreach(source IN LISTS test_srcs)
    kaldi_decoder_add_test(${source})
  endforeach()
endif()

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS kaldi-decoder-core DESTINATION lib)
endif()
