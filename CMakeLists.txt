cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(kaldi-decoder)

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

set(KALDI_DECODER_VERSION "0.2.4")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(KALDI_DECODER_RPATH_ORIGIN "$ORIGIN")
else()
  set(CMAKE_MACOSX_RPATH ON)
  set(KALDI_DECODER_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${KALDI_DECODER_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${KALDI_DECODER_RPATH_ORIGIN})

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)
option(KALDI_DECODER_ENABLE_TESTS "Whether to build tests" ON)
option(KALDI_DECODER_BUILD_PYTHON "Whether to build Python" ON)

if(BUILD_SHARED_LIBS AND MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NOT BUILD_SHARED_LIBS AND MSVC)
  # see https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html
  # https://stackoverflow.com/questions/14172856/compile-with-mt-instead-of-md-using-cmake
  if(MSVC)
      add_compile_options(
          $<$<CONFIG:>:/MT> #---------|
          $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
          $<$<CONFIG:Release>:/MT> #--|
      )
  endif()
endif()

if(KALDI_DECODER_BUILD_PYTHON)
  include(pybind11)
endif()

include(kaldifst)
include(eigen)

if(KALDI_DECODER_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()

add_subdirectory(kaldi-decoder)
