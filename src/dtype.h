// ----------------------------------------------------------------------------
//
// dtype.h -- enum and operations for tensor data types
//
// Copyright (C) 2019 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _DTYPE_H_
#define _DTYPE_H_

#include <cstdint>
#include <string>
#include <array>

#include "util.h"

using namespace std;

namespace npy {
    /** This enum represents the different types of tensor data that can be stored. */
    enum class data_type : std::uint8_t {
        /** 8 bit signed integer */
        INT8, 
        /** 8 bit unsigned integer */
        UINT8, 
        /** 16-bit signed integer (short) */
        INT16, 
        /** 16-bit unsigned integer (ushort) */
        UINT16,
        /** 32-bit signed integer (int) */ 
        INT32, 
        /** 32-bit unsigned integer (uint) */
        UINT32,
        /** 64-bit integer (long) */ 
        INT64, 
        /** 64-bit unsigned integer (long) */
        UINT64, 
        /** 32-bit floating point value (float) */
        FLOAT32,
        /** 64-bit floating point value (double) */ 
        FLOAT64 
    };

    /** Convert a data type and endianness to a NPY dtype string.
     *  \param dtype the data type
     *  \param endian the endianness. Defaults to the current endianness of the caller.
     *  \return the NPY dtype string
     */
    const std::string& to_dtype(data_type dtype, endian endian=endian::NATIVE);

    /** Converts from an NPY dtype string to a data type and endianness.
     *  \param dtype the NPY dtype string
     *  \return a pair of data type and endianness corresponding to the input
     */
    const std::pair<data_type, endian>& from_dtype(const std::string& dtype);
}

#endif