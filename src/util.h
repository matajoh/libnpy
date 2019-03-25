// ----------------------------------------------------------------------------
//
// util.h -- helpful utilities for the library
//
// Copyright (C) 2019 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _UTIL_H_
#define _UTIL_H_

#include <cstdint>

using namespace std;

namespace npy
{
/** Enumeration which represents a type of endianness */
enum class endian : std::uint8_t
{
    /** Indicates that the native endianness should be used. Native in this case means
         *  that of the hardware the program is currently running on.
         */
    NATIVE,
    /** Indicates the use of big-endian encoding */
    BIG,
    /** Indicates the use of little-endian encoding */
    LITTLE
};

/** This function will return the endianness of the current hardware. */
inline endian native_endian()
{
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    return endian_test.c[0] == 1 ? endian::BIG : endian::LITTLE;
};
} // namespace npy

#endif