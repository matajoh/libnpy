#include <cstdint>
#include <cstdio>
#include <ctime>

#include "libnpy_tests.h"

namespace {
const std::string TEMP_NPZ = "temp.npz";
}

namespace {
void _test(int &result, npy::compression_method_t compression_method) {
  std::string asset_name = "test.npz";
  std::string suffix = "";
  if (compression_method == npy::compression_method_t::DEFLATED) {
    asset_name = "test_compressed.npz";
    suffix = "_compressed";
  }

  std::string expected = test::read_asset(asset_name);

  {
    npy::npzfilewriter npz(TEMP_NPZ, compression_method, npy::endian_t::LITTLE);
    npz.write("color", test::test_tensor<std::uint8_t>({5, 5, 3}));
    npz.write("depth.npy", test::test_tensor<float>({5, 5}));
    npz.write("unicode.npy", test::test_tensor<std::wstring>({5, 2, 5}));
  }

  std::string actual = test::read_file(TEMP_NPZ);
  test::assert_equal(expected, actual, result, "npz_write" + suffix);

  std::filesystem::remove(TEMP_NPZ);
}

void _test_memory(int &result) {
  std::string asset_name = "test.npz";

  std::string expected = test::read_asset(asset_name);
  std::string actual;

  npy::npzstringwriter npz(npy::compression_method_t::STORED,
                           npy::endian_t::LITTLE);
  npz.write("color", test::test_tensor<std::uint8_t>({5, 5, 3}));
  npz.write("depth.npy", test::test_tensor<float>({5, 5}));
  npz.write("unicode.npy", test::test_tensor<std::wstring>({5, 2, 5}));
  npz.close();
  actual = npz.str();

  test::assert_equal(expected, actual, result, "npz_write_memory");
}
} // namespace

int test_npz_write() {
  int result = EXIT_SUCCESS;

  _test(result, npy::compression_method_t::STORED);
  _test(result, npy::compression_method_t::DEFLATED);
  _test_memory(result);

  return result;
}