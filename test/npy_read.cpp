#include "libnpy_tests.h"
#include "npy_read.h"


void test_read_unicode(int& result)
{
    std::vector<std::size_t> shape({5, 2, 5});
    npy::tensor<std::wstring> expected(shape);
    int i=0;
    for(auto& word : expected)
    {
        word = std::to_wstring(i);
        i += 1;
    }

    npy::tensor<std::wstring> actual = npy::load<std::wstring, npy::tensor>(test::asset_path("unicode.npy"));

    std::string tag = "unicode";
    test::assert_equal(to_dtype(expected.dtype()), to_dtype(actual.dtype()), result, tag + " dtype");
    test::assert_equal(expected.fortran_order(), actual.fortran_order(), result, tag + " fortran_order");
    test::assert_equal(expected.shape(), actual.shape(), result, tag + " shape");

    auto expected_it = expected.begin();
    auto actual_it = actual.begin();
    for (std::size_t i = 0; i < expected.size(); ++i, ++expected_it, ++actual_it)
    {
        if (*expected_it != *actual_it)
        {
            result = EXIT_FAILURE;
            std::wcout << std::wstring(tag.begin(), tag.end()) << " is incorrect: " << *actual_it << " != " << *expected_it << std::endl;
            break;
        }
    }
}

int test_npy_read()
{
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
    test_read_unicode(result);

    return result;
}