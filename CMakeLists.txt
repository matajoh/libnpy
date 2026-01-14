cmake_minimum_required( VERSION 3.15 )

# -------------------- Version --------------------------------

file( STRINGS "VERSION" LIBNPY_VERSION_FILE )

string( REPLACE "." ";" LIBNPY_VERSION_LIST ${LIBNPY_VERSION_FILE} )

list( GET LIBNPY_VERSION_LIST 0 LIBNPY_VERSION_MAJOR )

list( GET LIBNPY_VERSION_LIST 1 LIBNPY_VERSION_MINOR )

list( GET LIBNPY_VERSION_LIST 2 LIBNPY_VERSION_REVISION )

set( LIBNPY_VERSION ${LIBNPY_VERSION_MAJOR}.${LIBNPY_VERSION_MINOR}.${LIBNPY_VERSION_REVISION} )

message("Configure LIBNPY_VERSION at ${LIBNPY_VERSION}")

project( npy VERSION ${LIBNPY_VERSION} LANGUAGES CXX)

# -------------------- Options --------------------------------

option( LIBNPY_BUILD_TESTS "Specifies whether to build the tests" OFF )
option( LIBNPY_BUILD_DOCUMENTATION "Specifies whether to build the documentation for the API and XML" OFF )
set( LIBNPY_SANITIZE "" CACHE STRING "Argument to pass to sanitize (disabled by default)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------- Find packages --------------------------

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

find_program(CLANG_FORMAT NAMES clang-format-10 clang-format-14 clang-format-18 )

string(COMPARE EQUAL ${CLANG_FORMAT} "CLANG_FORMAT-NOTFOUND" CLANG_FORMAT_NOT_FOUND)
if(CLANG_FORMAT_NOT_FOUND)
  message("libnpy_format target not defined: no clang-format tool found")
else()
  file(GLOB ALL_SOURCE_FILES CONFIGURE_DEPENDS
        src/*.cpp
        src/*.h
        include/libnpy/*.h
        test/*.cpp
        test/*.h
        examples/**/*.cpp
  )

  add_custom_target(libnpy_format
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMAND ${CLANG_FORMAT}
                    -i
                    ${ALL_SOURCE_FILES})
endif()

# -------------------- Walk the subdirectories --------------------

add_subdirectory( src )

if( LIBNPY_BUILD_SAMPLES )
  add_subdirectory( samples )
endif()

if( LIBNPY_BUILD_DOCUMENTATION )
  find_package( Doxygen REQUIRED )
  add_subdirectory( doc )
endif()

target_include_directories(npy
  PUBLIC
    $<INSTALL_INTERFACE:build/native/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# -------------------- Testing ------------------------------------

if( LIBNPY_BUILD_TESTS )
  include( CTest )
  add_subdirectory( test )
endif()


# -------------------- INSTALL ------------------------------------

set(INSTALL_CONFIGDIR cmake)
set(INSTALL_LIBDIR lib)
set(INSTALL_INCLUDEDIR include)
set(LIBNPY_INSTALL_TARGETS npy)

install(TARGETS ${LIBNPY_INSTALL_TARGETS}
  EXPORT ${PROJECT_NAME}_Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/cmake
)

install(EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/cmake)

install(FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/cmake)

export(EXPORT ${PROJECT_NAME}_Targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::)

export(PACKAGE ${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDEDIR})
install(FILES 
  DESTINATION ${INSTALL_INCLUDEDIR}/npy
)

# -------------------- Package ------------------------------------

set( PROJECT_FILES
  README.md
  CHANGELOG.md
)

set( CPACK_SYSTEM_NAME ${SYSTEM_NAME} )
set( CPACK_PACKAGE_VERSION "${LIBNPY_VERSION}" )
set( CPACK_GENERATOR "ZIP" )
set( CPACK_SOURCE_GENERATOR "ZIP" )
set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0 )
include( CPack )