include_guard(GLOBAL)

function(download_kaldifst)
  # Already provided by a parent project (e.g. kaldi-hmm-gmm's ./cmake).
  if(TARGET kaldifst_core)
    return()
  endif()

  include(FetchContent)

  # NOTE: keep in sync with kaldi-hmm-gmm/cmake/kaldifst.cmake and the pip
  # requirement `kaldifst>=1.9.0`; v1.9.0+ provides the zero-copy interop
  # helpers (_vector_fst_addr/_borrow/_take) used by the python bindings.
  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.9.2.tar.gz")
  set(kaldifst_HASH "SHA256=8007cfde794f6c5e287d0e5e22753c1cf3503c9b02512fbcd6dc5d75e5b41fa2")

  # If you don't have access to the Internet,
  # please pre-download kaldifst
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-1.9.2.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldifst-1.9.2.tar.gz
    ${CMAKE_BINARY_DIR}/kaldifst-1.9.2.tar.gz
    /tmp/kaldifst-1.9.2.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.9.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldifst_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldifst_URL}" kaldifst_URL)
      message(STATUS "Found local downloaded kaldifst: ${kaldifst_URL}")
      break()
    endif()
  endforeach()

  set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldifst
    URL               ${kaldifst_URL}
    URL_HASH          ${kaldifst_HASH}
  )

  FetchContent_GetProperties(kaldifst)
  if(NOT kaldifst_POPULATED)
    message(STATUS "Downloading kaldifst from ${kaldifst_URL}")
    FetchContent_Populate(kaldifst)
  endif()
  message(STATUS "kaldifst is downloaded to ${kaldifst_SOURCE_DIR}")
  message(STATUS "kaldifst's binary dir is ${kaldifst_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake)

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR} EXCLUDE_FROM_ALL)

  # Hide the bundled openfst + kaldifst symbols unconditionally.  Their
  # FstRegister registry singletons are weak template statics: exported,
  # they coalesce process-wide with any other module bundling openfst
  # (kaldifst, kaldi-hmm-gmm), and the surviving registry may lack types
  # another module registered (vosk's olabel_lookahead/ngram graphs then
  # fail to read).  Hidden: each module keeps its own registry; FSTs
  # still cross module boundaries by raw pointer (python/csrc/fst-caster.h),
  # which does not rely on symbol sharing.  Contract unchanged: all
  # modules must bundle the same openfst sources/flags (same layouts).
  set_target_properties(kaldifst_core
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
  )
  foreach(_t fst fstscript fstlookahead fstngram)
    if(TARGET ${_t})
      set_target_properties(${_t}
        PROPERTIES
          POSITION_INDEPENDENT_CODE ON
          C_VISIBILITY_PRESET hidden
          CXX_VISIBILITY_PRESET hidden
      )
    endif()
  endforeach()

  if(_build_shared_libs_bak)
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(kaldifst_core
    PUBLIC
      ${kaldifst_SOURCE_DIR}/
  )

  set_target_properties(kaldifst_core PROPERTIES OUTPUT_NAME "kaldi-decoder-kaldi-fst-core")
endfunction()

download_kaldifst()
