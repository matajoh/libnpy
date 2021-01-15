cmake_minimum_required( VERSION 3.8 )

project( libnpy )

# -------------------- Version --------------------------------

file( STRINGS "VERSION" LIBNPY_VERSION_FILE )

string( REPLACE "." ";" LIBNPY_VERSION_LIST ${LIBNPY_VERSION_FILE} )

list( GET LIBNPY_VERSION_LIST 0 LIBNPY_VERSION_MAJOR )

list( GET LIBNPY_VERSION_LIST 1 LIBNPY_VERSION_MINOR )

list( GET LIBNPY_VERSION_LIST 2 LIBNPY_VERSION_REVISION )

set( LIBNPY_VERSION ${LIBNPY_VERSION_MAJOR}.${LIBNPY_VERSION_MINOR}.${LIBNPY_VERSION_REVISION} )

message("Configure LIBNPY_VERSION at ${LIBNPY_VERSION}")

# -------------------- Options --------------------------------

set( BUILD_DOCUMENTATION_DESC "Specifies whether to build the documentation for the API and XML" )
set( INCLUDE_CSHARP_DESC  "Specifies whether to build libnpy with C# bindings" )

set( BUILD_DOCUMENTATION ON  CACHE BOOL ${BUILD_DOCUMENTATION_DESC} )
set( INCLUDE_CSHARP      ON  CACHE BOOL ${INCLUDE_CSHARP_DESC} )

set( WARNINGS_AS_ERRORS OFF )

# ------------------- Detect the system -----------------------

if( APPLE )
  set( SYSTEM_NAME "Mac" )
elseif( UNIX )
  set( SYSTEM_NAME "Linux" )
  execute_process(COMMAND lsb_release -is OUTPUT_VARIABLE LSB_ID)
  execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE LSB_RELEASE)
  if( LSB_ID AND LSB_RELEASE )
    string(STRIP "${LSB_ID}" LSB_ID)
    string(STRIP "${LSB_RELEASE}" LSB_RELEASE)
    set( SYSTEM_NAME "${SYSTEM_NAME}-${LSB_ID}-${LSB_RELEASE}")
  elseif( EXISTS "/etc/debian_version")
    file( READ /etc/debian_version DEBIAN_VERSION )
    set( SYSTEM_NAME "${SYSTEM_NAME}-${DEBIAN_VERSION}")
  elseif( EXISTS "/etc/os-release")
    execute_process(COMMAND "sed" "-ne" "s/^ID=\"\\?\\([a-z]\\+\\)\"\\?$/\\1/p" "/etc/os-release" OUTPUT_VARIABLE OS_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "sed" "-ne" "s/^VERSION_ID=\"\\?\\([0-9\\.]\\+\\)\"\\?$/\\1/p" "/etc/os-release" OUTPUT_VARIABLE OS_RELEASE_VERSION_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    set( SYSTEM_NAME "${SYSTEM_NAME}-${OS_RELEASE_ID}-${OS_RELEASE_VERSION_ID}")
  elseif( EXISTS "/etc/redhat-release")
    set( SYSTEM_NAME "${SYSTEM_NAME}-Redhat")
  endif()
elseif( WIN32 )
  set( SYSTEM_NAME "Windows" )
endif()
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set( SYSTEM_NAME "${SYSTEM_NAME}-64bit" )
else()
  set( SYSTEM_NAME "${SYSTEM_NAME}-32bit" )
endif()

# -------------------- Find packages --------------------------

if ( WIN32 )
    enable_language( CSharp )
else()
    set( INCLUDE_CSHARP OFF )
endif()

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if( INCLUDE_CSHARP )
  find_package( SWIG REQUIRED )

  # Select the .NET architecture
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( CSHARP_PLATFORM "x64" CACHE STRING "C# target platform: x86, x64, anycpu, or itanium")
  else()
    set( CSHARP_PLATFORM "x86" CACHE STRING "C# target platform: x86, x64, anycpu, or itanium")
  endif()
endif()

find_package( Git REQUIRED )

if( BUILD_DOCUMENTATION )
  find_package( Doxygen REQUIRED )
endif()

# -------------------- Git hooks ------------------------------

file( GLOB GIT_HOOKS hooks/* )

foreach( hook_source ${GIT_HOOKS} )
  file(READ ${hook_source} content)
  string(REPLACE "<SOURCE_DIR>" "${CMAKE_SOURCE_DIR}" content ${content})
  get_filename_component(hook_name ${hook_source} NAME)
  get_filename_component(script_ext ${hook_source} EXT)
  if (${script_ext} STREQUAL ".sh")
	  string(REPLACE ${script_ext} "" hook_name ${hook_name})
	  string(REPLACE "_" "-" hook_name ${hook_name})
	  set( output_path "${CMAKE_SOURCE_DIR}/.git/hooks/${hook_name}")
	  file( GENERATE OUTPUT ${output_path} CONTENT ${content})
  else()
	# assume it is the Python hook script...
	  string(REPLACE "_" "-" hook_name ${hook_name})
	  set( output_path "${CMAKE_SOURCE_DIR}/.git/hooks/${hook_name}")
	  file( GENERATE OUTPUT ${output_path} CONTENT ${content})
  endif()
endforeach()

# -------------------- Build settings -----------------------------

# use C++11
if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
endif ()
 
if( MSVC )
  set( LIBNPY_CSHARP_DIR ${CMAKE_BINARY_DIR}/src/CSharpWrapper/$<CONFIG> )
  if( WARNINGS_AS_ERRORS )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX" )
  endif()
  # set the Windows target version to WinXP or later
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN32_WINNT=0x0501" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS" )
else()
  set( LIBNPY_CSHARP_DIR ${CMAKE_BINARY_DIR}/src/CSharpWrapper )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )
  if( WARNINGS_AS_ERRORS )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror" )
  else()
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations" ) # get a lot of warnings from Boost about auto_ptr
  endif()  
endif()

if( APPLE )
  add_definitions( -DAPPLE )
  set( CMAKE_MACOSX_RPATH ON )
endif()

# -------------------- Testing ------------------------------------

set( BUILDNAME ${CMAKE_SYSTEM} ) # including version numbers
if( MSVC )
  set( BUILDNAME "${BUILDNAME}_MSVC" )
elseif( CMAKE_COMPILER_IS_GNUCXX )
  set( BUILDNAME "${BUILDNAME}_gcc" )
else()
  set( BUILDNAME "${BUILDNAME}_${CMAKE_CXX_COMPILER}" )
endif()

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set( BUILDNAME "${BUILDNAME}_64bit" )
else()
  set( BUILDNAME "${BUILDNAME}_32bit" )
endif()

include( CTest )

# -------------------- Walk the subdirectories --------------------

add_subdirectory( src )
add_subdirectory( test )
add_subdirectory( samples )
if( BUILD_DOCUMENTATION )
  add_subdirectory( doc )
endif()

# -------------------- Package ------------------------------------

set( CPACK_SYSTEM_NAME ${SYSTEM_NAME} )
set( CPACK_PACKAGE_VERSION "${LIBNPY_VERSION}" )
set( CPACK_GENERATOR "ZIP" )
set( CPACK_SOURCE_GENERATOR "ZIP" )
include( CPack )
