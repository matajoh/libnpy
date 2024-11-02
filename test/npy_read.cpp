#include "npy_read.h"
#include "libnpy_tests.h"

int test_npy_read() {
  int result = EXIT_SUCCESS;

  test_read<std::uint8_t>(result, "uint8");
  test_read<std::uint8_t>(result, "uint8_fortran", true);
  test_read<std::int8_t>(result, "int8");
  test_read<std::uint16_t>(result, "uint16");
  test_read<std::int16_t>(result, "int16");
  test_read<std::uint32_t>(result, "uint32");
  test_read<std::int32_t>(result, "int32");
  test_read<std::int32_t>(result, "int32_big");
  test_read_scalar<std::int32_t>(result, "int32_scalar");
  test_read_array<std::int32_t>(result, "int32_array");
  test_read<std::uint64_t>(result, "uint64");
  test_read<std::int64_t>(result, "int64");
  test_read<float>(result, "float32");
  test_read<double>(result, "float64");
  test_read<std::wstring>(result, "unicode");

  return result;
}