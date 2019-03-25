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

#include <string>
#include <cstdint>

namespace zip {
    /** Deflate the bytes and return the compressed result.
     *  \param bytes the raw bytes
     *  \return the compressed bytes
     */
    std::string deflate(const std::string& bytes);

    /** Inflate the bytes and return the decompressed result.
     *  \param bytes the compressed bytes
     *  \return the raw bytes
     */
    std::string inflate(const std::string& bytes);

    /** Perform a fast CRC32 checksum of a set of bytes.
     *  \param bytes the bytes to check
     *  \return the CRC32 checksum
     */
    std::uint32_t crc32(const std::string& bytes);
}

#endif