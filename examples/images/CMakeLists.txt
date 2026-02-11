cmake_minimum_required(VERSION 3.15)

include(FetchContent)

project( npy_images VERSION 0.1.0 LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (DEFINED ENV{LIBNPY_REPO})
  set(LIBNPY_REPO $ENV{LIBNPY_REPO})
else ()
  set(LIBNPY_REPO "https://github.com/matajoh/libnpy/")
endif ()

if (DEFINED ENV{LIBNPY_TAG})
  set(LIBNPY_TAG $ENV{LIBNPY_TAG})
else ()
  set(LIBNPY_TAG "main")
endif()

if(LIBNPY_REPO STREQUAL LOCAL)
  set(FETCHCONTENT_SOURCE_DIR_LIBNPY "${CMAKE_SOURCE_DIR}/../..")
  message("LIBNPY_REPO=LOCAL: Using ${FETCHCONTENT_SOURCE_DIR_LIBNPY} as source directory")
endif()

FetchContent_Declare(
  libnpy
  GIT_REPOSITORY ${LIBNPY_REPO}
  GIT_TAG        ${LIBNPY_TAG}
)

FetchContent_MakeAvailable(libnpy)

add_executable( npy_images images.cpp )
target_link_libraries( npy_images npy::npy )

enable_testing()
add_test(
  NAME npy_images
  COMMAND npy_images
)