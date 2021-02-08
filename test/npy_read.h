#ifndef _NPY_READ_H_
#define _NPY_READ_H_

#include "libnpy_tests.h"
#include "npy/tensor.h"
#include "npy/npy.h"

template <typename T>
void test_read(int &result, const std::string &name, bool fortran_order = false)
{
    npy::tensor<T> expected = test::test_tensor<T>({5, 2, 5});
    if (fortran_order)
    {
        expected = test::test_fortran_tensor<T>();
    }

    npy::tensor<T> actual = npy::load<T, npy::tensor>(test::asset_path(name + ".npy"));
    test::assert_equal(expected, actual, result, "npy_read_" + name);
}

template <typename T>
void test_read_scalar(int &result, const std::string &name)
{
    npy::tensor<T> expected = test::test_tensor<T>({});
    *expected.data() = static_cast<T>(42);
    npy::tensor<T> actual = npy::load<T, npy::tensor>(test::asset_path(name + ".npy"));
    test::assert_equal(expected, actual, result, "npy_read_" + name);
}

#endif