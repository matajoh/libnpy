#include "libnpy_tests.h"
#include <complex>

int test_npy_write() {
  int result = EXIT_SUCCESS;

  std::string expected, actual;

  npy::endian_t endianness = npy::native_endian();
  std::string dtype =
      npy::to_dtype(npy::data_type_t::FLOAT32, npy::endian_t::BIG);

  expected = test::read_asset("uint8.npy");
  actual = test::npy_stream<std::uint8_t>();
  test::assert_equal(expected, actual, result, "npy_write_uint8");

  expected = test::read_asset("uint8_fortran.npy");
  actual = test::npy_fortran_stream<std::uint8_t>();
  test::assert_equal(expected, actual, result, "npy_write_uint8_fortran");

  expected = test::read_asset("int8.npy");
  actual = test::npy_stream<std::int8_t>();
  test::assert_equal(expected, actual, result, "npy_write_int8");

  expected = test::read_asset("uint16.npy");
  actual = test::npy_stream<std::uint16_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_uint16");

  expected = test::read_asset("int16.npy");
  actual = test::npy_stream<std::int16_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_int16");

  expected = test::read_asset("uint32.npy");
  actual = test::npy_stream<std::uint32_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_uint32");

  expected = test::read_asset("int32.npy");
  actual = test::npy_stream<std::int32_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_int32");

  expected = test::read_asset("int32_big.npy");
  actual = test::npy_stream<std::int32_t>(npy::endian_t::BIG);
  test::assert_equal(expected, actual, result, "npy_write_int32_big");

  expected = test::read_asset("int32_scalar.npy");
  actual = test::npy_scalar_stream<std::int32_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_int32_scalar");

  expected = test::read_asset("int32_array.npy");
  actual = test::npy_array_stream<std::int32_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_int32_array");

  expected = test::read_asset("uint64.npy");
  actual = test::npy_stream<std::uint64_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_uint64");

  expected = test::read_asset("int64.npy");
  actual = test::npy_stream<std::int64_t>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_int64");

  expected = test::read_asset("float32.npy");
  actual = test::npy_stream<float>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_float32");

  expected = test::read_asset("float64.npy");
  actual = test::npy_stream<double>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_float64");

  expected = test::read_asset("complex64.npy");
  actual = test::npy_stream<std::complex<float>>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_complex64");

  expected = test::read_asset("complex128.npy");
  actual = test::npy_stream<std::complex<double>>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_complex128");

  expected = test::read_asset("unicode.npy");
  actual = test::npy_stream<std::wstring>(npy::endian_t::LITTLE);
  test::assert_equal(expected, actual, result, "npy_write_unicode");

  expected = test::read_asset("bool.npy");
  actual = test::npy_stream<bool>();
  test::assert_equal(expected, actual, result, "npy_write_bool");

  return result;
};
