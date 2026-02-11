#ifndef _LIBNPY_TESTS_H_
#define _LIBNPY_TESTS_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "npy/npy.h"

int test_crc32();
int test_exceptions();
int test_npy_peek();
int test_npy_read();
int test_npy_write();
int test_npz_peek();
int test_npz_read();
int test_npz_write();
int test_tensor();
int test_custom_tensor();

namespace test {
template <typename T>
void assert_equal(const T &expected, const T &actual, int &result,
                  const std::string &tag) {
  if (expected != actual) {
    result = EXIT_FAILURE;
    std::cout << tag << " is incorrect: " << actual << " != " << expected
              << std::endl;
  }
}

template <typename T>
void assert_equal(const std::vector<T> &expected, const std::vector<T> &actual,
                  int &result, const std::string &tag) {
  assert_equal(expected.size(), actual.size(), result, tag + " size");
  if (result == EXIT_SUCCESS) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      assert_equal(expected[i], actual[i], result,
                   tag + "[" + std::to_string(i) + "]");
      if (result == EXIT_FAILURE) {
        break;
      }
    }
  }
}

template <typename T>
void assert_equal(const npy::tensor<T> &expected, const npy::tensor<T> &actual,
                  int &result, const std::string &tag) {
  assert_equal(expected.dtype(), actual.dtype(), result, tag + " dtype");
  assert_equal(expected.fortran_order(), actual.fortran_order(), result,
               tag + " fortran_order");
  assert_equal(expected.shape(), actual.shape(), result, tag + " shape");
  assert_equal(expected.values(), actual.values(), result, tag);
}

template <>
inline void assert_equal<std::string>(const std::string &expected,
                                      const std::string &actual, int &result,
                                      const std::string &tag) {
  assert_equal(expected.length(), actual.length(), result, tag + " length");
  if (result == EXIT_SUCCESS) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      int expected_val = static_cast<int>(expected[i]);
      int actual_val = static_cast<int>(actual[i]);
      assert_equal(expected_val, actual_val, result,
                   tag + "[" + std::to_string(i) + "]");
      if (result == EXIT_FAILURE) {
        break;
      }
    }
  }
}

template <>
inline void assert_equal<std::wstring>(const std::wstring &expected,
                                       const std::wstring &actual, int &result,
                                       const std::string &tag) {
  assert_equal(expected.length(), actual.length(), result, tag + " length");
  if (result == EXIT_SUCCESS) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      int expected_val = static_cast<int>(expected[i]);
      int actual_val = static_cast<int>(actual[i]);
      assert_equal(expected_val, actual_val, result,
                   tag + "[" + std::to_string(i) + "]");
      if (result == EXIT_FAILURE) {
        break;
      }
    }
  }
}

template <>
inline void assert_equal<npy::header_info>(const npy::header_info &expected,
                                           const npy::header_info &actual,
                                           int &result,
                                           const std::string &tag) {
  assert_equal(expected.dtype, actual.dtype, result, tag + " dtype");
  assert_equal(expected.endianness, actual.endianness, result,
               tag + " endianness");
  assert_equal(expected.fortran_order, actual.fortran_order, result,
               tag + " fortran_order");
  assert_equal(expected.shape, actual.shape, result, tag + " shape");
}

template <>
inline void assert_equal<npy::tensor<std::wstring>>(
    const npy::tensor<std::wstring> &expected,
    const npy::tensor<std::wstring> &actual, int &result,
    const std::string &tag) {
  assert_equal(expected.dtype(), actual.dtype(), result, tag + " dtype");
  assert_equal(expected.fortran_order(), actual.fortran_order(), result,
               tag + " fortran_order");
  assert_equal(expected.shape(), actual.shape(), result, tag + " shape");

  auto expected_it = expected.begin();
  auto actual_it = actual.begin();
  for (std::size_t i = 0; i < expected.size();
       ++i, ++expected_it, ++actual_it) {
    if (*expected_it != *actual_it) {
      result = EXIT_FAILURE;
      std::wcout << std::wstring(tag.begin(), tag.end())
                 << " is incorrect: " << *actual_it << " != " << *expected_it
                 << std::endl;
      break;
    }
  }
}

template <class EXCEPTION>
void assert_throws(void (*function)(), int &result, const std::string &tag) {
  try {
    function();
    result = EXIT_FAILURE;
    std::cout << tag << " did not throw an exception" << std::endl;
  } catch (const EXCEPTION &) {
    std::cout << tag << " expected exception thrown" << std::endl;
    return;
  } catch (const std::exception &e) {
#ifndef __APPLE__ // macos sometimes falls through erroneously.
    result = EXIT_FAILURE;
#endif
    std::cout << tag << " threw unexpected exception: " << e.what()
              << std::endl;
  }
}

template <class EXCEPTION, typename Arg>
void assert_throws(void (*function)(Arg), Arg arg, int &result,
                   const std::string &tag) {
  try {
    function(arg);
    result = EXIT_FAILURE;
    std::cout << tag << " did not throw an exception" << std::endl;
  } catch (const EXCEPTION &) {
    std::cout << tag << " expected exception thrown" << std::endl;
    return;
  } catch (const std::exception &e) {
#ifndef __APPLE__ // macos sometimes falls through erroneously.
    result = EXIT_FAILURE;
#endif
    std::cout << tag << " threw unexpected exception: " << e.what()
              << std::endl;
  }
}

template <typename T>
npy::tensor<T> test_tensor(const std::vector<size_t> &shape) {
  npy::tensor<T> tensor(shape);
  std::vector<T> values(tensor.size());
  auto curr = values.begin();
  for (int i = 0; curr < values.end(); ++i, ++curr) {
    *curr = static_cast<T>(i);
  }
  tensor.copy_from(values);

  return tensor;
};

template <>
inline npy::tensor<std::wstring> test_tensor(const std::vector<size_t> &shape) {
  npy::tensor<std::wstring> tensor(shape);
  int i = 0;
  for (auto &word : tensor) {
    word = std::to_wstring(i);
    i += 1;
  }

  return tensor;
}

template <>
inline npy::tensor<npy::boolean> test_tensor(const std::vector<size_t> &shape) {
  npy::tensor<npy::boolean> tensor(shape);
  int i = 0;
  for (auto it = tensor.begin(); it != tensor.end(); ++it, ++i) {
    *it = (i % 2) == 1; // Alternating false, true, false, true, ...
  }

  return tensor;
}

template <typename T> npy::tensor<T> test_fortran_tensor() {
  std::vector<int> values = {0,  10, 20, 30, 40, 5,  15, 25, 35, 45, 1,  11, 21,
                             31, 41, 6,  16, 26, 36, 46, 2,  12, 22, 32, 42, 7,
                             17, 27, 37, 47, 3,  13, 23, 33, 43, 8,  18, 28, 38,
                             48, 4,  14, 24, 34, 44, 9,  19, 29, 39, 49};
  npy::tensor<T> tensor({5, 2, 5}, true);
  auto dst = tensor.data();
  auto src = values.begin();
  for (; dst < tensor.data() + tensor.size(); ++src, ++dst) {
    *dst = static_cast<T>(*src);
  }

  return tensor;
}

template <> inline npy::tensor<std::wstring> test_fortran_tensor() {
  std::vector<int> values = {0,  10, 20, 30, 40, 5,  15, 25, 35, 45, 1,  11, 21,
                             31, 41, 6,  16, 26, 36, 46, 2,  12, 22, 32, 42, 7,
                             17, 27, 37, 47, 3,  13, 23, 33, 43, 8,  18, 28, 38,
                             48, 4,  14, 24, 34, 44, 9,  19, 29, 39, 49};
  npy::tensor<std::wstring> tensor({5, 2, 5}, true);
  auto dst = tensor.data();
  auto src = values.begin();
  for (; dst < tensor.data() + tensor.size(); ++src, ++dst) {
    *dst = std::to_wstring(*src);
  }

  return tensor;
}

template <> inline npy::tensor<npy::boolean> test_fortran_tensor() {
  std::vector<int> values = {0,  10, 20, 30, 40, 5,  15, 25, 35, 45, 1,  11, 21,
                             31, 41, 6,  16, 26, 36, 46, 2,  12, 22, 32, 42, 7,
                             17, 27, 37, 47, 3,  13, 23, 33, 43, 8,  18, 28, 38,
                             48, 4,  14, 24, 34, 44, 9,  19, 29, 39, 49};
  npy::tensor<npy::boolean> tensor({5, 2, 5}, true);
  auto dst = tensor.begin();
  auto src = values.begin();
  for (; dst != tensor.end(); ++src, ++dst) {
    *dst = (*src % 2) == 1;
  }

  return tensor;
}

template <typename T>
std::string npy_stream(npy::endian_t endianness = npy::endian_t::NATIVE) {
  std::ostringstream actual_stream;
  npy::tensor<T> tensor = test_tensor<T>({5, 2, 5});
  npy::save<T, npy::tensor>(actual_stream, tensor, endianness);
  return actual_stream.str();
}

template <typename T>
std::string
npy_scalar_stream(npy::endian_t endianness = npy::endian_t::NATIVE) {
  std::ostringstream actual_stream;
  npy::tensor<T> tensor = test_tensor<T>({});
  *tensor.data() = static_cast<T>(42);
  npy::save(actual_stream, tensor, endianness);
  return actual_stream.str();
}

template <typename T>
std::string npy_array_stream(npy::endian_t endianness = npy::endian_t::NATIVE) {
  std::ostringstream actual_stream;
  npy::tensor<T> tensor = test_tensor<T>({25});
  npy::save(actual_stream, tensor, endianness);
  return actual_stream.str();
}

template <typename T>
std::string
npy_fortran_stream(npy::endian_t endianness = npy::endian_t::NATIVE) {
  std::ostringstream actual_stream;
  npy::tensor<T> tensor = test_fortran_tensor<T>();
  npy::save(actual_stream, tensor, endianness);
  return actual_stream.str();
}

std::string read_file(const std::string &path);
std::string read_asset(const std::string &filename);
std::string asset_path(const std::string &filename);
std::string path_join(const std::vector<std::string> &parts);
} // namespace test

#endif