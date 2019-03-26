set( CPP_TEST_SOURCES
   libnpy_tests.cpp
   libnpy_tests.h
)

set( TESTS
   exceptions
   npy_write
   npy_read
   npz_write
   npz_read
   tensor
)

foreach( test ${TESTS} )
   list( APPEND CPP_TEST_SOURCES "${test}.cpp" )
endforeach()

set( LIBNPYTESTS_INCLUDE_FOLDERS
   ${CMAKE_CURRENT_SOURCE_DIR}/../src
   ${CMAKE_CURRENT_SOURCE_DIR}
)
include_directories( ${LIBNPYTESTS_INCLUDE_FOLDERS} )

set( TEST_DRIVER libnpy_tests )
add_executable( ${TEST_DRIVER} ${CPP_TEST_SOURCES} )
target_link_libraries( ${TEST_DRIVER} npy )

foreach( test ${TESTS} )
add_test( NAME ${test}
        COMMAND ${TEST_DRIVER} ${test}
        WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY} )
endforeach()


if( INCLUDE_CSHARP )
  add_subdirectory( CSharpTests )
endif()
