set( CSHARP_TEST_SOURCES
  test_npy_read.cs
  test_npy_write.cs 
  test_npz_read.cs
  test_npz_write.cs
)

if( WIN32 )
  set( TEST_FOLDER ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG> )
else()
  set( TEST_FOLDER ${CMAKE_CURRENT_BINARY_DIR} )
endif()

foreach( file ${CSHARP_TEST_SOURCES} )
  get_filename_component( test_name "${file}" NAME_WE )
  set( test_name "CSharpTests_${test_name}" )  

  add_executable( ${test_name} ${file} Test.cs )
  target_link_libraries( ${test_name} NumpyIO ${SWIG_MODULE_NumpyIONative_REAL_NAME} )

  # copy the NumpyIO dynamic libraries to the test build folder
  add_custom_command( TARGET ${test_name} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${LIBNPY_CSHARP_DIR}/NumpyIO.dll ${TEST_FOLDER}/NumpyIO.dll )
  add_custom_command( TARGET ${test_name} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${LIBNPY_CSHARP_DIR}/${NUMPYIO_NATIVE} ${TEST_FOLDER}/${NUMPYIO_NATIVE} )

  if( NOT WIN32 )
    add_test( NAME ${test_name}
              COMMAND ${CSHARP_INTERPRETER} ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.exe
              WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY} )
  else()
    add_test( NAME ${test_name} 
              COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.exe
              WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY} )
  endif()

endforeach()