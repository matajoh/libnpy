cmake_minimum_required( VERSION 3.13...3.16 FATAL_ERROR )

# -------------------- Version --------------------------------

file( STRINGS "VERSION" LIBNPY_VERSION_FILE )

string( REPLACE "." ";" LIBNPY_VERSION_LIST ${LIBNPY_VERSION_FILE} )

list( GET LIBNPY_VERSION_LIST 0 LIBNPY_VERSION_MAJOR )

list( GET LIBNPY_VERSION_LIST 1 LIBNPY_VERSION_MINOR )

list( GET LIBNPY_VERSION_LIST 2 LIBNPY_VERSION_REVISION )

set( LIBNPY_VERSION ${LIBNPY_VERSION_MAJOR}.${LIBNPY_VERSION_MINOR}.${LIBNPY_VERSION_REVISION} )

message("Configure LIBNPY_VERSION at ${LIBNPY_VERSION}")

project( libnpy VERSION ${LIBNPY_VERSION} LANGUAGES CXX)

# -------------------- Options --------------------------------

option( LIBNPY_BUILD_TESTS "Specifies whether to build the tests" OFF )
option( LIBNPY_BUILD_SAMPLES "Specifies whether to build the samples" OFF )
option( LIBNPY_BUILD_DOCUMENTATION "Specifies whether to build the documentation for the API and XML" OFF )
option( LIBNPY_INCLUDE_CSHARP  "Specifies whether to build libnpy with C# bindings" OFF )
set( LIBNPY_SANITIZE "" CACHE STRING "Argument to pass to sanitize (disabled by default)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------- Find packages --------------------------

if ( WIN32 AND INCLUDE_CSHARP )
    enable_language( CSharp )
else()
    set( INCLUDE_CSHARP OFF )
endif()

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if( INCLUDE_CSHARP )
  find_package( SWIG REQUIRED )

  # Select the .NET architecture
  set( CSHARP_PLATFORM_DESC "C# target platform: x86, x64, anycpu, or itanium" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( CSHARP_PLATFORM "x64" CACHE STRING ${CSHARP_PLATFORM_DESC})
  else()
    set( CSHARP_PLATFORM "x86" CACHE STRING ${CSHARP_PLATFORM_DESC})
  endif()
endif()

if( BUILD_DOCUMENTATION )
  find_package( Doxygen REQUIRED )
endif()

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
        examples/*.cpp
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
  add_subdirectory( doc )
endif()

if( LIBNPY_INCLUDE_CSHARP )
  add_subdirectory( CSharpWrapper )
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
  if( MSVC )
    set( LIBNPY_CSHARP_DIR ${CMAKE_BINARY_DIR}/CSharpWrapper/$<CONFIG> )
  endif()

  include( CTest )
  add_subdirectory( test )
endif()


# -------------------- Build settings -----------------------------

# use C++11
target_compile_features(npy PRIVATE cxx_std_11)

# -------------------- INSTALL ------------------------------------

set(INSTALL_CONFIGDIR "cmake")

install(TARGETS npy 
  EXPORT npy-targets
  ARCHIVE DESTINATION "build/native/lib"
  LIBRARY DESTINATION "build/native/lib"
)

install(DIRECTORY include/ DESTINATION "build/native/include")

install(EXPORT npy-targets
  FILE
    npyTargets.cmake
  NAMESPACE
    npy::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/npyConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/npyConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/npyConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/npyConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/npyConfigVersion.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

export(EXPORT npy-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/npyTargets.cmake
  NAMESPACE npy::
)

export(PACKAGE npy)

# -------------------- Package ------------------------------------

set( PROJECT_FILES
  README.md
  CHANGELOG.md
)

# copy these files into the root of the distribution zip
install( FILES ${PROJECT_FILES} DESTINATION "." )

if( MSVC )
  # NuGet files
  set( LIBNPY_NUGET_NAME "npy-${SYSTEM_TOOLKIT}-${SYSTEM_BITS}-${CMAKE_BUILD_TYPE}" CACHE STRING "npy NuGet Name" FORCE )
  file( READ RELEASE_NOTES LIBNPY_RELEASE_NOTES )

  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/nuget/template.nuspec.in" "${CMAKE_CURRENT_BINARY_DIR}/nuget/${LIBNPY_NUGET_NAME}.nuspec" @ONLY )
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/nuget/template.targets.in"
                "${CMAKE_CURRENT_BINARY_DIR}/nuget/build/native/${LIBNPY_NUGET_NAME}.targets" @ONLY )
else()
  set( CPACK_SYSTEM_NAME ${SYSTEM_NAME} )
  set( CPACK_PACKAGE_VERSION "${LIBNPY_VERSION}" )
  set( CPACK_GENERATOR "ZIP" )
  set( CPACK_SOURCE_GENERATOR "ZIP" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0 )
  include( CPack )
endif()
