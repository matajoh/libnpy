// ----------------------------------------------------------------------------
//
// core.h -- core types, enums and functions used by the library
//
// Copyright (C) 2021 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _CORE_H_
#define _CORE_H_

#include <cstdint>
#include <iostream>
#include <string>

namespace npy {
/** Enumeration which represents a type of endianness */
enum class endian_t : char {
  /** Indicates that the native endianness should be used. Native in this case
   * means that of the hardware the program is currently running on.
   */
  NATIVE,
  /** Indicates the use of big-endian encoding */
  BIG,
  /** Indicates the use of little-endian encoding */
  LITTLE
};

/** This function will return the endianness of the current hardware. */
inline endian_t native_endian() {
  union {
    std::uint32_t i;
    char c[4];
  } endian_test = {0x01020304};

  return endian_test.c[0] == 1 ? endian_t::BIG : endian_t::LITTLE;
};

/** This enum represents the different types of tensor data that can be stored.
 */
enum class data_type_t : char {
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
  FLOAT64,
  /** Unicode string (std::wstring) */
  UNICODE_STRING
};

/** Convert a data type and endianness to a NPY dtype string.
 *  \param dtype the data type
 *  \param endian the endianness. Defaults to the current endianness of the
 * caller. \return the NPY dtype string
 */
const std::string &to_dtype(data_type_t dtype,
                            endian_t endian = endian_t::NATIVE);

/** Converts from an NPY dtype string to a data type and endianness.
 *  \param dtype the NPY dtype string
 *  \return a pair of data type and endianness corresponding to the input
 */
const std::pair<data_type_t, endian_t> &from_dtype(const std::string &dtype);

typedef std::basic_istringstream<char> imemstream;
typedef std::basic_ostringstream<char> omemstream;

std::ostream &operator<<(std::ostream &os, const endian_t &obj);
std::ostream &operator<<(std::ostream &os, const data_type_t &obj);

} // namespace npy

#endif