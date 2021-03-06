#include "libnpy_tests.h"
#include "npy/npz.h"

namespace
{
void _test(int &result, const std::string &filename, bool compressed)
{
    auto expected_color = test::test_tensor<std::uint8_t>({5, 5, 3});
    auto expected_depth = test::test_tensor<float>({5, 5});
    auto expected_unicode = test::test_tensor<std::wstring>({5, 2, 5});

    npy::inpzstream stream(test::asset_path(filename));
    auto actual_color = stream.read<std::uint8_t, npy::tensor>("color.npy");
    auto actual_depth = stream.read<float, npy::tensor>("depth");
    auto actual_unicode = stream.read<std::wstring, npy::tensor>("unicode");

    std::string suffix = compressed ? "_compressed" : "";
    test::assert_equal(expected_color, actual_color, result, "npz_read_color" + suffix);
    test::assert_equal(expected_depth, actual_depth, result, "npz_read_depth" + suffix);
    test::assert_equal(expected_unicode, actual_unicode, result, "npz_read_unicode" + suffix);
}
} // namespace

int test_npz_read()
{
    int result = EXIT_SUCCESS;

    _test(result, "test.npz", false);
    _test(result, "test_compressed.npz", true);

    return result;
}