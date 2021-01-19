set( CPP_TEST_SOURCES
   libnpy_tests.cpp
   libnpy_tests.h
)

set( TESTS
   crc32
   exceptions
   memstream
   npy_peek
   npy_read
   npy_write
   npz_peek
   npz_read
   npz_write
   tensor
)

foreach( test ${TESTS} )
   list( APPEND CPP_TEST_SOURCES "${test}.cpp" )
endforeach()

set( TEST_DRIVER libnpy_tests )
add_executable( ${TEST_DRIVER} ${CPP_TEST_SOURCES} )
target_link_libraries( ${TEST_DRIVER} npy::npy )
target_include_directories( ${TEST_DRIVER} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src)

foreach( test ${TESTS} )
add_test( NAME ${test}
        COMMAND ${TEST_DRIVER} ${test}
        WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY} )
endforeach()


if( INCLUDE_CSHARP )
  add_subdirectory( CSharpTests )
endif()
