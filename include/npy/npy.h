// ----------------------------------------------------------------------------
//
// npy.h -- methods for reading and writing the numpy lib (NPY) format. The
//          implementation is based upon the description available at:
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
//
// Copyright (C) 2021 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _NPY_H_
#define _NPY_H_

#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cassert>

#include "core.h"

using namespace std;

const int STATIC_HEADER_LENGTH = 10;

namespace npy
{
/** Class representing the header info for an NPY file */
struct header_info
{
    /** Constructor.
     *  \param dictionary a Python-encoded dictionary containing the header information
     */
    explicit header_info(const std::string &dictionary);

    /** Constructor */
    header_info(data_type_t dtype,
                npy::endian_t endianness,
                bool fortran_order,
                const std::vector<size_t> &shape);

    /** The data type of the NPY file */
    data_type_t dtype;

    /** The endianness of the data in the NPY file */
    npy::endian_t endianness;

    /** Whether the values in the tensor are stored in FORTRAN, or column major, order */
    bool fortran_order;

    /** A vector of values indicating the shape of each dimension of the tensor. */
    std::vector<size_t> shape;

    /** Value used to indicate the maximum length of an element (used by Unicode strings) */
    std::size_t max_element_length;
};

/** Writes an NPY header to the provided stream.
 *  \param output the output stream
 *  \param dtype the NPY-encoded dtype string (includes data type and endianness)
 *  \param fortran_order whether the data is encoded in FORTRAN (i.e. column major) order
 *  \param shape a sequence of values indicating the shape of each dimension of the tensor
 *  \sa npy::to_dtype
 */
template <typename CHAR>
void write_npy_header(std::basic_ostream<CHAR> &output,
                      const std::string &dtype,
                      bool fortran_order,
                      const std::vector<size_t> &shape)
{
    std::ostringstream buff;
    buff << "{'descr': '" << dtype;
    buff << "', 'fortran_order': " << (fortran_order ? "True" : "False");
    buff << ", 'shape': (";
    for (auto dim = shape.begin(); dim < shape.end(); ++dim)
    {
        buff << *dim;
        if (dim < shape.end() - 1)
        {
            buff << ", ";
        }
    }

    if(shape.size() == 1)
    {
        buff << ",";
    }

    buff << "), }";
    std::string dictionary = buff.str();
    auto dict_length = dictionary.size() + 1;
    std::string end = "\n";
    auto header_length = dict_length + STATIC_HEADER_LENGTH;
    if (header_length % 64 != 0)
    {
        header_length = ((header_length / 64) + 1) * 64;
        dict_length = header_length - STATIC_HEADER_LENGTH;
        end = std::string(dict_length - dictionary.length(), ' ');
        end.back() = '\n';
    }

    std::uint8_t header[STATIC_HEADER_LENGTH] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00,
                                                 static_cast<std::uint8_t>(dict_length),
                                                 0x00};
    output.write(reinterpret_cast<const CHAR *>(header), STATIC_HEADER_LENGTH);
    output.write(reinterpret_cast<const CHAR *>(dictionary.data()), dictionary.length());
    output.write(reinterpret_cast<const CHAR *>(end.data()), end.length());
}

template<typename T, typename CHAR>
void copy_to(const T* data_ptr, std::size_t num_elements, std::basic_ostream<CHAR>& output, npy::endian_t endianness)
{
    if (endianness == npy::endian_t::NATIVE || endianness == native_endian())
    {
        output.write(reinterpret_cast<const CHAR *>(data_ptr), num_elements * sizeof(T));
    }
    else
    {
        CHAR buffer[sizeof(T)];
        for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr)
        {
            const CHAR *start = reinterpret_cast<const CHAR *>(curr);
            std::reverse_copy(start, start + sizeof(T), buffer);
            output.write(buffer, sizeof(T));
        }
    }    
}

/** Saves a tensor to the provided stream.
 *  \tparam T the data type
 *  \tparam TENSOR the tensor type.
 *  \param output the output stream
 *  \param tensor the tensor
 *  \param endianness the endianness to use in saving the tensor
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR,
          typename CHAR,
          std::enable_if_t<!std::is_same<std::wstring, T>::value, int> = 42>
void save(std::basic_ostream<CHAR> &output,
          const TENSOR<T> &tensor,
          endian_t endianness = npy::endian_t::NATIVE)
{
    auto dtype = to_dtype(tensor.dtype(), endianness);
    write_npy_header(output, dtype, tensor.fortran_order(), tensor.shape());
    copy_to(tensor.data(), tensor.size(), output, endianness);
};

/** Saves a unicode string tensor to the provided stream.
 *  \tparam TENSOR the tensor type.
 *  \param output the output stream
 *  \param tensor the tensor
 *  \param endianness the endianness to use in saving the tensor
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR,
          typename CHAR,
          std::enable_if_t<std::is_same<std::wstring, T>::value, int> = 42>
void save(std::basic_ostream<CHAR> &output,
          const TENSOR<std::wstring> &tensor,
          endian_t endianness = npy::endian_t::NATIVE)
{
    std::size_t max_length = 0;
    for(const auto& element : tensor)
    {
        if(element.size() > max_length)
        {
            max_length = element.size();
        }
    }

    if(endianness == npy::endian_t::NATIVE)
    {
        endianness = native_endian();
    }

    std::string dtype = ">U" + std::to_string(max_length);
    if(endianness == npy::endian_t::LITTLE)
    {
        dtype = "<U" + std::to_string(max_length);
    }

    write_npy_header(output, dtype, tensor.fortran_order(), tensor.shape());

    std::vector<std::int32_t> unicode(tensor.size() * max_length, 0);
    auto word_start = unicode.begin();
    for(const auto& element : tensor)
    {
        auto char_it = word_start;
        for(const auto& wchar : element)
        {
            *char_it = static_cast<std::int32_t>(wchar);
            char_it += 1;
        }

        word_start += max_length;
    }

    copy_to(unicode.data(), unicode.size(), output, endianness);
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

    save<T, TENSOR, char>(output, tensor, endianness);
};

/** Read an NPY header from the provided stream.
 *  \param input the input stream
 *  \return the header information
 */
template <typename CHAR>
header_info read_npy_header(std::basic_istream<CHAR> &input)
{
    std::uint8_t header[STATIC_HEADER_LENGTH];
    input.read(reinterpret_cast<CHAR *>(header), STATIC_HEADER_LENGTH);
    assert(header[0] == 0x93);
    assert(header[1] == 'N');
    assert(header[2] == 'U');
    assert(header[3] == 'M');
    assert(header[4] == 'P');
    assert(header[5] == 'Y');
    size_t dict_length = 0;
    if (header[6] == 0x01 && header[7] == 0x00)
    {
        dict_length = header[8] | (header[9] << 8);
    }
    else if (header[6] == 0x02 && header[7] == 0x00)
    {
        std::uint8_t extra[2];
        input.read(reinterpret_cast<CHAR *>(extra), 2);
        dict_length = header[8] | (header[9] << 8) | (extra[0] << 16) | (extra[1] << 24);
    }

    std::vector<CHAR> buffer(dict_length);
    input.read(buffer.data(), dict_length);
    std::string dictionary(buffer.begin(), buffer.end());
    return header_info(dictionary);
}

template <typename T, typename CHAR>
void copy_to(std::basic_istream<CHAR> &input, T* data_ptr, std::size_t num_elements, npy::endian_t endianness)
{
    if (endianness == npy::endian_t::NATIVE || endianness == native_endian())
    {
        CHAR *start = reinterpret_cast<CHAR *>(data_ptr);
        input.read(start, num_elements * sizeof(T));
    }
    else
    {
        CHAR buffer[sizeof(T)];
        for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr)
        {
            input.read(buffer, sizeof(T));
            CHAR *start = reinterpret_cast<CHAR *>(curr);
            std::reverse_copy(buffer, buffer + sizeof(T), start);
        }
    }    
}

/** Loads a tensor in NPY format from the provided stream. The type of the tensor
 *  must match the data to be read.
 *  \tparam T the data type
 *  \tparam TENSOR the tensor type
 *  \param input the input stream
 *  \return an object of type TENSOR<T> read from the stream
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR,
          typename CHAR,
          std::enable_if_t<!std::is_same<std::wstring, T>::value, int> = 42>
TENSOR<T> load(std::basic_istream<CHAR> &input)
{
    header_info info = read_npy_header(input);
    TENSOR<T> tensor(info.shape, info.fortran_order);
    if (info.dtype != tensor.dtype())
    {
        throw std::logic_error("requested dtype does not match stream's dtype");
    }

    copy_to(input, tensor.data(), tensor.size(), info.endianness);
    return tensor;
}


/** Loads a unicode string tensor in NPY format from the provided stream. The type of the tensor
 *  must match the data to be read.
 *  \tparam T the data type
 *  \tparam TENSOR the tensor type
 *  \param input the input stream
 *  \return an object of type TENSOR<T> read from the stream
 *  \sa npy::tensor
 */
template <typename T,
          template <typename> class TENSOR,
          typename CHAR,
          std::enable_if_t<std::is_same<std::wstring, T>::value, int> = 42>
TENSOR<T> load(std::basic_istream<CHAR> &input)
{
    header_info info = read_npy_header(input);
    TENSOR<T> tensor(info.shape, info.fortran_order);
    if (info.dtype != tensor.dtype())
    {
        throw std::logic_error("requested dtype does not match stream's dtype");
    }

    std::vector<std::int32_t> unicode(tensor.size() * info.max_element_length, 0);
    copy_to(input, unicode.data(), unicode.size(), info.endianness);

    auto word_start = unicode.begin();
    for(auto& element : tensor)
    {
        auto char_it = word_start;
        for(std::size_t i=0; i<info.max_element_length && *char_it > 0; ++i, ++char_it)
        {
            element.push_back(static_cast<wchar_t>(*char_it));
        }

        word_start += info.max_element_length;
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

/** Return the header information for an NPY file.
 *  \param input the input stream containing the NPY-encoded bytes
 *  \return the NPY header information
 */
template <typename CHAR>
header_info peek(std::basic_istream<CHAR> &input)
{
    return read_npy_header(input);
}

/** Return the header information for an NPY file.
 *  \param path the path to the NPY file on disk
 *  \return the NPY header information
 */
header_info peek(const std::string &path);

} // namespace npy

#endif