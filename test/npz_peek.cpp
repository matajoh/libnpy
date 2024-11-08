#include "libnpy_tests.h"
#include "npy/npz.h"

namespace {
void _test(int &result, const std::string &filename, bool compressed) {
  npy::header_info expected_color(npy::data_type_t::UINT8,
                                  npy::endian_t::NATIVE, false, {5, 5, 3});
  npy::header_info expected_depth(npy::data_type_t::FLOAT32,
                                  npy::endian_t::LITTLE, false, {5, 5});

  npy::inpzstream stream(test::asset_path(filename));
  const auto &keys = stream.keys();

  test::assert_equal(keys[0], std::string("color.npy"), result,
                     "npz_keys_incorrect");
  test::assert_equal(keys[1], std::string("depth.npy"), result,
                     "npz_keys_incorrect");
  test::assert_equal(keys[2], std::string("unicode.npy"), result,
                     "npz_keys_incorrect");
  test::assert_equal(false, stream.contains("not_there.npy"), result,
                     "npz_contains_missing");
  test::assert_equal(true, stream.contains("color.npy"), result,
                     "npz_contains_color");
  test::assert_equal(true, stream.contains("depth.npy"), result,
                     "npz_contains_depth");

  npy::header_info actual_color = stream.peek("color.npy");
  npy::header_info actual_depth = stream.peek("depth.npy");

  std::string suffix = compressed ? "_compressed" : "";
  test::assert_equal(expected_color, actual_color, result,
                     "npz_peek_color" + suffix);
  test::assert_equal(expected_depth, actual_depth, result,
                     "npz_peek_depth" + suffix);
}
} // namespace

int test_npz_peek() {
  int result = EXIT_SUCCESS;

  _test(result, "test.npz", false);
  _test(result, "test_compressed.npz", true);

  return result;
}