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

void _test_large(int &result, const std::string &filename, bool compressed)
{
    auto expected_int = test::test_tensor<std::int32_t>({200, 5, 1000});
    auto expected_float = test::test_tensor<float>({1000, 5, 20, 10});

    npy::inpzstream stream(test::asset_path(filename));
    auto actual_int = stream.read<std::int32_t, npy::tensor>("test_int");
    auto actual_float = stream.read<float, npy::tensor>("test_float");

    std::string suffix = compressed ? "_compressed" : "";
    test::assert_equal(expected_int, actual_int, result, "npz_read_large_int" + suffix);
    test::assert_equal(expected_float, actual_float, result, "npz_read_large_float" + suffix);
}
} // namespace

int test_npz_read()
{
    int result = EXIT_SUCCESS;

    _test(result, "test.npz", false);
    _test(result, "test_compressed.npz", true);
    _test_large(result, "test_large.npz", false);
    _test_large(result, "test_large_compressed.npz", true);

    return result;
}