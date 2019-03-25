#ifndef _NPY_READ_H_
#define _NPY_READ_H_

#include "libnpy_tests.h"
#include "tensor.h"
#include "npy.h"

template <typename T>
void test_read(int &result, const std::string &name, bool fortran_order = false)
{
    npy::tensor<T> expected = test::test_tensor<T>({5, 2, 5});
    if (fortran_order)
    {
        expected = test::test_fortran_tensor<T>();
    }

    std::string path = test::path_join({"assets", "test", name + ".npy"});
    npy::tensor<T> actual = npy::load<T, npy::tensor>(path);
    test::assert_equal(expected, actual, result, "npy_read_" + name);
}

#endif