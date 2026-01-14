cmake_minimum_required(VERSION 3.15)

include(FetchContent)

project( npy_images VERSION 0.1.0 LANGUAGES CXX )

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

FetchContent_Declare(
  libnpy
  GIT_REPOSITORY ${LIBNPY_REPO}
  GIT_TAG        ${LIBNPY_TAG}
)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)

FetchContent_MakeAvailable(libnpy)

FetchContent_GetProperties(Eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(Eigen)

  add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable( npy_custom custom.cpp )
target_link_libraries( npy_custom npy::npy Eigen3::Eigen)

enable_testing()
add_test(
  NAME npy_custom
  COMMAND npy_custom
)
set_property(TEST npy_custom PROPERTY LABELS TESTLABEL npy)
