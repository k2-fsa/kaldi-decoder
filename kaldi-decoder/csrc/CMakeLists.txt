# PROJECT_SOURCE_DIR (not CMAKE_SOURCE_DIR): correct both standalone
# and when built as a subproject (e.g. inside kaldi-hmm-gmm).
include_directories(${PROJECT_SOURCE_DIR})

# Please keep the source files alphabetically sorted
set(srcs
  decodable-ctc.cc
  decodable-matrix.cc
  decoder-wrappers.cc
  eigen.cc
  faster-decoder.cc
  lattice-faster-decoder.cc
  lattice-simple-decoder.cc
  simple-decoder.cc
)

# Always static build
add_library(kaldi-decoder-core STATIC ${srcs})

if(NOT WIN32)
  set_target_properties(kaldi-decoder-core
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
  )
endif()

target_link_libraries(kaldi-decoder-core PUBLIC kaldifst_core)
target_link_libraries(kaldi-decoder-core PUBLIC Eigen3::Eigen)

if(KALDI_DECODER_ENABLE_TESTS)
  set(test_srcs
    api-coverage-test.cc
    decodable-matrix-test.cc
    decoder-wrappers-test.cc
    eigen-test.cc
    faster-decoder-test.cc
    hash-list-test.cc
    lattice-faster-decoder-test.cc
    lattice-simple-decoder-test.cc
    simple-decoder-test.cc
  )

  function(kaldi_decoder_add_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name "${name}")
    add_executable(${target_name} ${source})
    target_link_libraries(${target_name}
      PRIVATE
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

# ---- clang-tidy (top-level project only) ------------------------------------
# Sources to lint with `make check` (clang-tidy).  Keep this list in sync
# with scripts/check_style_clang_format.sh.  Don't pass --config-file:
# clang-tidy picks up the repo-root .clang-tidy automatically.
# Only lint files that are actually compiled (they have compile commands).
set(srcs_to_check)
foreach(source IN LISTS srcs test_srcs)
  list(APPEND srcs_to_check ${CMAKE_CURRENT_SOURCE_DIR}/${source})
endforeach()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

# The python bindings are compiled into the extension module.
file(GLOB _python_srcs ${PROJECT_SOURCE_DIR}/kaldi-decoder/python/csrc/*.cc)
list(APPEND srcs_to_check ${_python_srcs})

# For clang-tidy.  Requires CMAKE_EXPORT_COMPILE_COMMANDS (set at the
# top level) so that compile_commands.json exists in the build dir.
#
# On macOS, pass the SDK path so that the system headers (<cmath> etc.)
# are found; without it clang-tidy reports bogus parse errors.
set(_tidy_extra_args)
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE _tidy_sdk
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND _tidy_extra_args "--extra-arg=-isysroot${_tidy_sdk}")
endif()

add_custom_target(
  clang-tidy-check
  clang-tidy ${_tidy_extra_args} -p ${CMAKE_BINARY_DIR}/compile_commands.json
             ${srcs_to_check}
  DEPENDS ${srcs})

add_custom_target(check DEPENDS clang-tidy-check)

endif()  # top-level project only
