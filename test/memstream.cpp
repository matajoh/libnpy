#include <algorithm>

#include "libnpy_tests.h"

namespace {
const size_t SIZE = 50;

void test_read(int &result) {
  std::vector<char> values(SIZE);
  std::iota(values.begin(), values.end(), 0);
  std::string expected(values.begin(), values.end());

  npy::imemstream stream(expected);
  stream.read(values.data(), SIZE);
  std::string actual(values.begin(), values.end());

  test::assert_equal(expected, actual, result, "memstream_test_copy_read");

  stream = npy::imemstream(std::move(expected));
  std::fill(actual.begin(), actual.end(), 0);
  stream.read(actual.data(), SIZE);

  expected = std::move(stream.str());

  test::assert_equal(expected, actual, result, "memstream_test_move_read");
}

void test_write(int &result) {
  std::vector<char> values(SIZE);
  std::iota(values.begin(), values.end(), 0);
  std::string expected(values.begin(), values.end());

  npy::omemstream stream;
  stream.write(expected.data(), SIZE);

  std::string actual = stream.str();
  test::assert_equal(expected, actual, result, "memstream_test_copy_write");

  std::fill(actual.begin(), actual.end(), 0);
  stream = npy::omemstream(std::move(actual));
  stream.write(expected.data(), SIZE);
  actual = std::move(stream.str());

  test::assert_equal(expected, actual, result, "memstream_test_move_write");
}
} // namespace

int test_memstream() {
  int result = EXIT_SUCCESS;

  test_read(result);

  return result;
}