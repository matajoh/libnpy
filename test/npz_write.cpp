#include <sstream>
#include <memory>
#include <ctime>
#include <cstdint>
#include <cstdio>

#include "libnpy_tests.h"
#include "npz.h"

namespace
{
const char *TEMP_NPZ = "temp.npz";
}

namespace
{
void _test(int &result, npy::compression_method compression_method)
{
    std::string asset_name = "test.npz";
    std::string suffix = "";
    if (compression_method == npy::compression_method::DEFLATED)
    {
        asset_name = "test_compressed.npz";
        suffix = "_compressed";
    }

    std::string expected = test::read_asset(asset_name);

    {
        npy::onpzstream npz(TEMP_NPZ, compression_method, npy::endian_t::LITTLE);
        npz.write("color.npy", test::test_tensor<std::uint8_t>({5, 5, 3}));
        npz.write("depth.npy", test::test_tensor<float>({5, 5}));
    }

    std::string actual = test::read_file(TEMP_NPZ);
    test::assert_equal(expected, actual, result, "npz_write" + suffix);

    std::remove(TEMP_NPZ);
}
} // namespace

int test_npz_write()
{
    int result = EXIT_SUCCESS;

    _test(result, npy::compression_method::STORED);
    _test(result, npy::compression_method::DEFLATED);

    return result;
}