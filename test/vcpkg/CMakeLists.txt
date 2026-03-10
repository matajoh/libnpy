cmake_minimum_required(VERSION 3.15)

project(vcpkg_test VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(npy CONFIG REQUIRED)

add_executable(vcpkg_test main.cpp)
target_link_libraries(vcpkg_test PRIVATE npy::npy)

enable_testing()
add_test(NAME vcpkg_test COMMAND vcpkg_test)
