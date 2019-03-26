// ----------------------------------------------------------------------------
//
// npy.h -- methods for reading and writing the numpy lib (NPY) format. The
//          implementation is based upon the description available at:
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
//
// Copyright (C) 2019 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _NPY_H_
#define _NPY_H_

#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>

#include "core.h"

using namespace std;

namespace npy
{
/** Class representing the header info for an NPY file */
struct header_info
{
    /** Constructor.
     *  \param dictionary a Python-encoded dictionary containing the header information
     */
    header_info(const std::string &dictionary);

    /** The data type of the NPY file */
    data_type_t dtype;

    /** The endianness of the data in the NPY file */
    npy::endian_t endianness;

    /** Whether the values in the tensor are stored in FORTRAN, or column major, order */
    bool fortran_order;

    /** A vector of values indicating the shape of each dimension of the tensor. */
    std::vector<size_t> shape;
};

/** Writes an NPY header to the provided stream.
 *  \param output the output stream
 *  \param dtype the NPY-encoded dtype string (includes data type and endianness)
 *  \param fortran_order whether the data is encoded in FORTRAN (i.e. column major) order
 *  \param shape a sequence of values indicating the shape of each dimension of the tensor
 *  \sa npy::to_dtype
 */
void write_npy_header(std::ostream &output,
                      const std::string &dtype,
                      bool fortran_order,
                      const std::vector<size_t> &shape);

/** Saves a tensor to the provided stream.
 *  \tparam T the data type
 *  \tparam TENSOR the tensor type.
 *  \param output the output stream
 *  \param tensor the tensor
 *  \param endianness the endianness to use in saving the tensor
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR>
void save(std::ostream &output,
          const TENSOR<T> &tensor,
          endian_t endianness = npy::endian_t::NATIVE)
{
    auto dtype = to_dtype(tensor.dtype(), endianness);
    write_npy_header(output, dtype, tensor.fortran_order(), tensor.shape());

    if (endianness == npy::endian_t::NATIVE ||
        endianness == native_endian() ||
        dtype[0] == '|')
    {
        output.write(reinterpret_cast<const char *>(tensor.data()), tensor.size() * sizeof(T));
    }
    else
    {
        char buffer[sizeof(T)];
        for (auto curr = tensor.data(); curr < tensor.data() + tensor.size(); ++curr)
        {
            const char *start = reinterpret_cast<const char *>(curr);
            std::reverse_copy(start, start + sizeof(T), buffer);
            output.write(buffer, sizeof(T));
        }
    }
};

/** Saves a tensor to the provided location on disk.
 *  \tparam T the data type
 *  \tparam TENSOR the tensor type.
 *  \param path a path to a valid location on disk
 *  \param tensor the tensor
 *  \param endianness the endianness to use in saving the tensor
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR>
void save(const std::string &path,
          const TENSOR<T> &tensor,
          endian_t endianness = npy::endian_t::NATIVE)
{
    std::ofstream output(path, std::ios::out | std::ios::binary);
    if (!output.is_open())
    {
        throw std::invalid_argument("path");
    }

    save(output, tensor, endianness);
};

/** Read an NPY header from the provided stream.
 *  \param input the input stream
 *  \return the header information
 */
header_info read_npy_header(std::istream &input);

/** Loads a tensor in NPY format from the provided stream. The type of the tensor
 *  must match the data to be read.
 *  \tparam T the data type
 *  \tparam TENSOR the tensor type
 *  \param input the input stream
 *  \return an object of type TENSOR<T> read from the stream
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR>
TENSOR<T> load(std::istream &input)
{
    header_info info = read_npy_header(input);
    TENSOR<T> tensor(info.shape, info.fortran_order);
    if (info.dtype != tensor.dtype())
    {
        throw std::logic_error("requested dtype does not match stream's dtype");
    }

    if (info.endianness == npy::endian_t::NATIVE || info.endianness == native_endian())
    {
        char *start = reinterpret_cast<char *>(tensor.data());
        input.read(start, tensor.size() * sizeof(T));
    }
    else
    {
        char buffer[sizeof(T)];
        for (auto curr = tensor.data(); curr < tensor.data() + tensor.size(); ++curr)
        {
            input.read(buffer, sizeof(T));
            char *start = reinterpret_cast<char *>(curr);
            std::reverse_copy(buffer, buffer + sizeof(T), start);
        }
    }

    return tensor;
}

/** Loads a tensor in NPY format from the specified location on the disk. The type of the tensor
 *  must match the data to be read.
 *  \tparam T the data type
 *  \tparam TENSOR the tensor type
 *  \param path a valid location on the disk
 *  \return an object of type TENSOR<T> read from the stream
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR>
TENSOR<T> load(const std::string &path)
{
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input.is_open())
    {
        throw std::invalid_argument("path");
    }

    return load<T, TENSOR>(input);
}
} // namespace npy

#endif