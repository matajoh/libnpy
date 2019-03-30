// ----------------------------------------------------------------------------
//
// zip.h -- simple wrapper around ZLIB
//
// Copyright (C) 2019 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _ZIP_H_
#define _ZIP_H_

#include <vector>
#include <iostream>
#include <cstdint>

namespace npy
{
/** Deflate the bytes and return the compressed result.
 *  \param bytes the raw bytes
 *  \return the compressed bytes
 */
std::vector<std::uint8_t> deflate(std::vector<std::uint8_t> &&bytes);

/** Inflate the bytes and return the decompressed result.
 *  \param bytes the compressed bytes
 *  \return the raw bytes
 */
std::vector<std::uint8_t> inflate(std::vector<std::uint8_t> &&bytes);

/** Perform a fast CRC32 checksum of a set of bytes.
 *  \param bytes the bytes to check
 *  \return the CRC32 checksum
 */
std::uint32_t crc32(const std::vector<std::uint8_t> &bytes);
} // namespace npy

#endif