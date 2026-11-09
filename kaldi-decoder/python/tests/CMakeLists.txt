# Python tests for the kaldi_decoder package.  Registration mirrors
# ../kaldi-hmm-gmm/kaldi-hmm-gmm/python/tests/CMakeLists.txt: the tests run
# through `cmake -E env`, NOT ctest's ENVIRONMENT property — the property is a
# CMake list, so ";"-separated PYTHONPATH entries (Windows) get split into
# bogus entries ($<SEMICOLON> does not help: ctest re-splits after evaluating
# the genex).  `cmake -E env` keeps the whole assignment as one argument.

function(kaldi_decoder_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

  get_filename_component(kd_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  # The extension module lives in ${kd_path}/kaldi_decoder/lib/ (built there
  # by the _kaldi_decoder target); tests import the package from ${kd_path}.
  # PYTHONPATH entries are separated by ":" on posix and ";" on Windows.
  if(WIN32)
    set(_pathsep ";")
  else()
    set(_pathsep ":")
  endif()
  set(_pythonpath "${kd_path}${_pathsep}$ENV{PYTHONPATH}")

  add_test(NAME ${name}
    COMMAND
      "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_pythonpath}"
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  # Tests exit 77 when an optional dependency is unavailable -> ctest "skipped".
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  test_decodable_itf.py
  test_decodable_matrix.py
  test_decoders.py
  test_fst_interop.py
  test_lifetime.py
)

foreach(source IN LISTS py_test_files)
  kaldi_decoder_add_py_test(${source})
endforeach()
