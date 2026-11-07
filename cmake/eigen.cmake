include_guard(GLOBAL)

function(download_eigen)
  # Already provided by a parent project (e.g. kaldi-hmm-gmm's ./cmake).
  if(TARGET Eigen3::Eigen)
    return()
  endif()

  include(FetchContent)

  # NOTE: keep in sync with kaldi-hmm-gmm/cmake/eigen.cmake (5.0.1) so a
  # process never carries two Eigen versions.
  set(eigen_URL  "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz")
  set(eigen_HASH "SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec")

  # If you don't have access to the Internet,
  # please pre-download eigen
  set(possible_file_locations
    $ENV{HOME}/Downloads/eigen-5.0.1.tar.gz
    ${CMAKE_SOURCE_DIR}/eigen-5.0.1.tar.gz
    ${CMAKE_BINARY_DIR}/eigen-5.0.1.tar.gz
    /tmp/eigen-5.0.1.tar.gz
    /star-fj/fangjun/download/github/eigen-5.0.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(eigen_URL  "${f}")
      file(TO_CMAKE_PATH "${eigen_URL}" eigen_URL)
      break()
    endif()
  endforeach()

  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)

  message(STATUS "Downloading eigen ${eigen_URL}")
  FetchContent_Declare(eigen
    EXCLUDE_FROM_ALL
    URL               ${eigen_URL}
    URL_HASH          ${eigen_HASH}
  )
  FetchContent_MakeAvailable(eigen)
  message(STATUS "eigen is downloaded to ${eigen_SOURCE_DIR}")
  message(STATUS "eigen's binary dir is ${eigen_BINARY_DIR}")
endfunction()

download_eigen()

