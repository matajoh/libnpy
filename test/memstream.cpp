#include <algorithm>

#include "libnpy_tests.h"
#include "core.h"

namespace {
    const size_t SIZE = 50;

    void test_read(int& result)
    {
        std::vector<std::uint8_t> expected(SIZE);
        std::iota(expected.begin(), expected.end(), 0);
        
        npy::imemstream stream(expected);
        std::vector<std::uint8_t> actual(SIZE);
        stream.read(actual.data(), SIZE);

        test::assert_equal(expected, actual, result, "memstream_test_copy_read");

        stream = npy::imemstream(std::move(expected));
        std::fill(actual.begin(), actual.end(), 0);
        stream.read(actual.data(), SIZE);

        expected = std::move(stream.buf());

        test::assert_equal(expected, actual, result, "memstream_test_move_read");        
    }

    void test_write(int& result)
    {
        std::vector<std::uint8_t> expected(SIZE);
        std::iota(expected.begin(), expected.end(), SIZE);

        npy::omemstream stream;
        stream.write(expected.data(), SIZE);

        std::vector<std::uint8_t> actual = stream.buf();
        test::assert_equal(expected, actual, result, "memstream_test_copy_write");

        std::fill(actual.begin(), actual.end(), 0);
        stream = npy::omemstream(std::move(actual));
        stream.write(expected.data(), SIZE);
        actual = std::move(stream.buf());

        test::assert_equal(expected, actual, result, "memstream_test_move_write");
    }
}

int test_memstream()
{
    int result = EXIT_SUCCESS;

    test_read(result);

    return result;
}