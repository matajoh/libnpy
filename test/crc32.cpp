#include <fstream>
#include <sstream>
#include <cstdint>

#include "libnpy_tests.h"
#include "zip.h"

const static int BUF_SIZE = 4096;

int test_crc32()
{
    int result = EXIT_SUCCESS;
    std::ifstream stream(test::asset_path("float32.npy"), std::ios_base::in | std::ios_base::binary);
    char buffer[BUF_SIZE];
    std::vector<std::uint8_t> bytes;
    while(stream.good())
    {
        stream.read(buffer, BUF_SIZE);
        std::copy(buffer, buffer + stream.gcount(), std::back_inserter(bytes));
    }

    int actual = npy::npy_crc32(bytes);
    int expected = 928602993;
    test::assert_equal(expected, actual, result, "crc32");
    return result;
}