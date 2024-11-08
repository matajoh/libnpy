// ----------------------------------------------------------------------------
//
// npz.h -- methods for reading and writing the numpy archive (NPZ) file
//          format. The implementation here is based upon the PKZIP Application
//          note: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT.
//
// Copyright (C) 2021 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _NPZ_H_
#define _NPZ_H_

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core.h"
#include "npy.h"
#include "tensor.h"

namespace npy {
/** Enumeration indicating the compression method to use for data in the NPZ
 * archive. */
enum class compression_method_t : std::uint16_t {
  /** Store the data with no compression */
  STORED = 0,
  /** Use the DEFLATE algorithm to compress the data */
  DEFLATED = 8
};

/** Struct representing a file in the NPZ archive. */
struct file_entry {
  /** The name of the file */
  std::string filename;
  /** The CRC32 checksum of the uncompressed data */
  std::uint32_t crc32;
  /** The size of the compressed data */
  std::uint64_t compressed_size;
  /** The size of the uncompressed data */
  std::uint64_t uncompressed_size;
  /** The method used to compress the data */
  std::uint16_t compression_method;
  /** The offset of the file in the archive */
  std::uint64_t offset;

  /** Check if this entry matches another entry
   *  \param other the other entry
   *  \return if these entries match
   */
  bool check(const file_entry &other) const;
};

/** Class representing an output stream for an NPZ archive file */
class onpzstream {
public:
  /** Constructor
   *  \param output the output stream to write to
   *  \param compression how the entries should be compressed
   *  \param endianness the endianness to use in writing the entries
   */   
  onpzstream(const std::shared_ptr<std::ostream> &output,
             compression_method_t compression = compression_method_t::STORED,
             endian_t endianness = npy::endian_t::NATIVE);

  /** Constructor.
   *  \param path the path to the file on disk
   *  \param compression how the entries should be compressed
   *  \param endianness the endianness to use in writing the entries
   */
  onpzstream(const std::string &path,
             compression_method_t compression = compression_method_t::STORED,
             endian_t endianness = npy::endian_t::NATIVE);

  /** Whether the underlying stream has successfully been opened. */
  bool is_open() const;

  /** Closes this stream. This will write the directory and close
   *  the underlying stream as well. */
  void close();

  /** Write a tensor to the NPZ archive.
   *  \tparam T the data type
   *  \tparam TENSOR the tensor type
   *  \param filename the name of the file in the archive
   *  \param tensor the tensor to write
   */
  template <typename T, template <typename> class TENSOR>
  void write(const std::string &filename, const TENSOR<T> &tensor) {
    if (m_closed) {
      throw std::logic_error("Stream is closed");
    }

    omemstream output;
    save<T, TENSOR, omemstream::char_type>(output, tensor, m_endianness);

    std::string suffix = ".npy";
    std::string name = filename;
    if (name.size() < 4 ||
        !std::equal(suffix.rbegin(), suffix.rend(), name.rbegin())) {
      name += ".npy";
    }

    write_file(name, output.str());
  }

  /** Write a tensor to the NPZ archive.
   *  \tparam T the data type
   *  \param filename the name of the file in the archive
   *  \param tensor the tensor to write
   */
  template <typename T>
  void write(const std::string &filename, const tensor<T> &tensor) {
    write<T, npy::tensor>(filename, tensor);
  }

  /** Destructor. This will call
   *  \link npy::onpzstream::close \endlink, if it has not been called already.
   */
  ~onpzstream();

private:
  /** Write a file to the stream.
   *  \param filename the name of the file
   *  \param bytes the file data
   */
  void write_file(const std::string &filename, std::string &&bytes);

  bool m_closed;
  std::shared_ptr<std::ostream> m_output;
  compression_method_t m_compression_method;
  endian_t m_endianness;
  std::vector<file_entry> m_entries;
};

/** Class representing an input stream from an NPZ archive file */
class inpzstream {
public:
  /** Constructor.
   *  \param stream the input stream to read from
   */
  inpzstream(const std::shared_ptr<std::istream> &stream);

  /** Constructor.
   *  \param path the path to the NPZ file on the disk
   */
  inpzstream(const std::string &path);

  /** Whether the underlying stream has successfully been opened. */
  bool is_open() const;

  /** Closes the underlying stream. */
  void close();

  /** The keys of the tensors in the NPZ */
  const std::vector<std::string> &keys() const;

  /** Returns whether this NPZ contains the specified tensor
   *  \param filename the name of the tensor in the archive
   *  \return whether the tensor is in the archive
   */
  bool contains(const std::string &filename);

  /** Returns the header for a specified tensor.
   *  \param filename the name of the tensor in the archive
   *  \return the header for the tensor
   */
  header_info peek(const std::string &filename);

  /** Read a tensor from the archive. This method will thrown an exception if
   *  the tensor does not exist, or if the data type of the tensor does not
   * match the template type. \tparam T the data type \tparam TENSOR the tensor
   * type \param filename the name of the tensor in the archive. \return a
   * instance of TENSOR<T> read from the archive. \sa npy::tensor
   */
  template <typename T, template <typename> class TENSOR>
  TENSOR<T> read(const std::string &filename) {
    imemstream stream(read_file(filename));
    return load<T, TENSOR>(stream);
  }

  /** Read a tensor from the archive. This method will thrown an exception if
   *  the tensor does not exist, or if the data type of the tensor does not
   * match. \tparam T the data type \param filename the name of the tensor in
   * the archive. \return a instance of tensor<T> read from the archive.
   */
  template <typename T> tensor<T> read(const std::string &filename) {
    return read<T, npy::tensor>(filename);
  }

private:
  /** Reads the bytes for a file from the archive.
   *  \param filename the name of the file
   *  \return the raw file bytes
   */
  std::string read_file(const std::string &filename);

  /** Read all entries from the directory. */
  void read_entries();

  std::shared_ptr<std::istream> m_input;
  std::map<std::string, file_entry> m_entries;
  std::vector<std::string> m_keys;
};
} // namespace npy

#endif