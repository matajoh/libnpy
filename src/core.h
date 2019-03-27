// ----------------------------------------------------------------------------
//
// core.h -- core types, enums and functions used by the library
//
// Copyright (C) 2019 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _CORE_H_
#define _CORE_H_

#include <cstdint>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

namespace npy
{
/** Enumeration which represents a type of endianness */
enum class endian_t : std::uint8_t
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
inline endian_t native_endian()
{
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    return endian_test.c[0] == 1 ? endian_t::BIG : endian_t::LITTLE;
};

/** This enum represents the different types of tensor data that can be stored. */
enum class data_type_t : std::uint8_t
{
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
const std::string &to_dtype(data_type_t dtype, endian_t endian = endian_t::NATIVE);

/** Converts from an NPY dtype string to a data type and endianness.
 *  \param dtype the NPY dtype string
 *  \return a pair of data type and endianness corresponding to the input
 */
const std::pair<data_type_t, endian_t> &from_dtype(const std::string &dtype);

class membuf : public std::basic_streambuf<std::uint8_t>
{
  public:
    membuf();
    membuf(size_t n);
    membuf(const std::vector<std::uint8_t> &buffer);
    membuf(std::vector<std::uint8_t> &&buffer);

    std::vector<std::uint8_t> &buf();
    const std::vector<std::uint8_t> &buf() const;

  protected:
    membuf *setbuf(std::uint8_t *s, std::streamsize n) override;
    pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override;
    std::streamsize showmanyc() override;
    std::streamsize xsgetn(std::uint8_t *s, std::streamsize n) override;
    int_type underflow() override;
    int_type pbackfail(int_type c = traits_type::eof()) override;
    std::streamsize xsputn(const std::uint8_t *s, std::streamsize n) override;
    int_type overflow(int_type c = traits_type::eof()) override;

  private:
    std::vector<std::uint8_t> m_buffer;
    std::vector<std::uint8_t>::iterator m_posg;
    std::vector<std::uint8_t>::iterator m_posp;
};

class imemstream : public std::basic_istream<std::uint8_t>
{
  public:
    imemstream(const std::vector<std::uint8_t> &buffer);
    imemstream(std::vector<std::uint8_t> &&buffer);

    std::vector<std::uint8_t> &buf();
    const std::vector<std::uint8_t> &buf() const;

  private:
    membuf m_buffer;
};

class omemstream : public std::basic_ostream<std::uint8_t>
{
  public:
    omemstream();
    omemstream(std::vector<std::uint8_t> &&buffer);
    omemstream(std::streamsize capacity);

    std::vector<std::uint8_t> &buf();
    const std::vector<std::uint8_t> &buf() const;

  private:
    membuf m_buffer;
};

} // namespace npy

#endif