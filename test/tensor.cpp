#include <cstdint>
#include <cstdio>

#include "libnpy_tests.h"
#include "tensor.h"

namespace
{
const char *TEMP_NPY = "temp.npy";
}

int test_tensor()
{
    int result = EXIT_SUCCESS;

    npy::tensor<std::uint8_t> fortran({3, 4, 5}, true);
    std::uint8_t value = 0;
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            for (size_t k = 0; k < 5; ++k, ++value)
            {
                fortran({i, j, k}) = value;
            }
        }
    }

    fortran.save(TEMP_NPY);

    npy::tensor<std::uint8_t> from_file(TEMP_NPY);
    npy::tensor<std::uint8_t> standard(from_file.shape(), false);
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            for (size_t k = 0; k < 5; ++k)
            {
                standard({i, j, k}) = fortran({i, j, k});
            }
        }
    }

    for (std::uint8_t i = 0; i < 60; ++i)
    {
        test::assert_equal(i, standard.values()[i], result, "tensor read/write");
    }

    std::remove(TEMP_NPY);

    return result;
};
