#include "libnpy_tests.h"
#include "npy/npy.h"

namespace {
void test_peek(int &result, const std::string &tag, npy::data_type_t data_type,
               npy::endian_t endianness = npy::endian_t::LITTLE,
               bool fortran_order = false) {
  npy::header_info expected = {data_type, endianness, fortran_order, {5, 2, 5}};
  npy::header_info actual = npy::peek(test::asset_path(tag + ".npy"));
  test::assert_equal(expected, actual, result, tag);
}
} // namespace

int test_npy_peek() {
  int result = EXIT_SUCCESS;

  test_peek(result, "uint8", npy::data_type_t::UINT8, npy::endian_t::NATIVE);
  test_peek(result, "uint8_fortran", npy::data_type_t::UINT8,
            npy::endian_t::NATIVE, true);
  test_peek(result, "int8", npy::data_type_t::INT8, npy::endian_t::NATIVE);
  test_peek(result, "uint16", npy::data_type_t::UINT16);
  test_peek(result, "int16", npy::data_type_t::INT16);
  test_peek(result, "uint32", npy::data_type_t::UINT32);
  test_peek(result, "int32", npy::data_type_t::INT32);
  test_peek(result, "int32_big", npy::data_type_t::INT32, npy::endian_t::BIG);
  test_peek(result, "uint64", npy::data_type_t::UINT64);
  test_peek(result, "int64", npy::data_type_t::INT64);
  test_peek(result, "float32", npy::data_type_t::FLOAT32);
  test_peek(result, "float64", npy::data_type_t::FLOAT64);

  return result;
}