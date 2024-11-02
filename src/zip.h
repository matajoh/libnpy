// ----------------------------------------------------------------------------
//
// zip.h -- simple wrapper around miniz
//
// Copyright (C) 2021 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _ZIP_H_
#define _ZIP_H_

#include <cstdint>
#include <string>

namespace npy {
/** Deflate the bytes and return the compressed result.
 *  \param bytes the raw bytes
 *  \return the compressed bytes
 */
std::string npy_deflate(std::string &&bytes);

/** Inflate the bytes and return the decompressed result.
 *  \param bytes the compressed bytes
 *  \return the raw bytes
 */
std::string npy_inflate(std::string &&bytes);

/** Perform a fast CRC32 checksum of a set of bytes.
 *  \param bytes the bytes to check
 *  \return the CRC32 checksum
 */
std::uint32_t npy_crc32(const std::string &bytes);
} // namespace npy

#endif