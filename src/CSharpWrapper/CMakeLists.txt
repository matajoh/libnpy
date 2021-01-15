include( ${SWIG_USE_FILE} )

execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set_source_files_properties ( NumpyIONative.i PROPERTIES CPLUSPLUS ON )

# CSharp version requirements: http://msdn.microsoft.com/en-us/library/system.reflection.assemblyversionattribute.aspx
# major.minor[.build[.revision]] where all components are 16-bit unsigned integers

set(NUMPYIO_VERSION_CSHARP_AssemblyVersion "${LIBNPY_VERSION_MAJOR}.${LIBNPY_VERSION_MINOR}.${GIT_BRANCH}.${GIT_COMMIT_HASH}")

  # Make sure the nested directory structure exists
set(CSHARP_SOURCE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/swig CACHE INTERNAL "")
file(MAKE_DIRECTORY ${CSHARP_SOURCE_DIRECTORY})

  # Create swig target
set(CMAKE_SWIG_OUTDIR ${CSHARP_SOURCE_DIRECTORY})

set(CMAKE_SWIG_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR} -namespace \"NumpyIO\" ${CMAKE_SWIG_GLOBAL_FLAGS} ${CMAKE_SWIG_FLAGS})

SET_SOURCE_FILES_PROPERTIES(NumpyIONative.i PROPERTIES SWIG_FLAGS "-includeall")

SWIG_ADD_LIBRARY( NumpyIONative 
    LANGUAGE csharp
    TYPE SHARED 
    SOURCES NumpyIONative.i 
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SWIG_LINK_LIBRARIES( ${SWIG_MODULE_NumpyIONative_REAL_NAME} npy )

if( UNIX )
  set_target_properties(${SWIG_MODULE_NumpyIONative_REAL_NAME} PROPERTIES PREFIX "lib" SUFFIX ".so")
  set( NUMPYIO_NATIVE libNumpyIONative.so CACHE INTERNAL "The NumpyIO built library" )
else()
  set_target_properties(${SWIG_MODULE_NumpyIONative_REAL_NAME} PROPERTIES SUFFIX ".dll")
  set( NUMPYIO_NATIVE NumpyIONative.dll CACHE INTERNAL "The NumpyIONative built library" )
endif()

# Configure AssemblyInfo.cs
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/AssemblyInfo.cs.in
    ${CSHARP_SOURCE_DIRECTORY}/AssemblyInfo.cs
    @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Tensor.cs
    ${CSHARP_SOURCE_DIRECTORY}/Tensor.cs
    COPYONLY
)

# build the SWIG CSharp files into a wrapper library
FILE(GLOB SWIG_CSHARP_SOURCES ${CSHARP_SOURCE_DIRECTORY}/*.cs)

foreach( source ${SWIG_CSHARP_SOURCES} ) 
    set_source_files_properties(${source} PROPERTIES GENERATED TRUE)
endforeach(source)

# Add managed wrapper
add_library(
    NumpyIO
    SHARED
    ${SWIG_CSHARP_SOURCES}
)

target_compile_options( NumpyIO PUBLIC "/unsafe" )
target_link_libraries( NumpyIO ${SWIG_MODULE_NumpyIONative_REAL_NAME} )

install( TARGETS NumpyIO ${SWIG_MODULE_NumpyIONative_REAL_NAME} RUNTIME DESTINATION "lib/net472")