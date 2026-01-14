/// ----------------------------------------------------------------------------
///
/// @file npy.h
/// @brief Definitions for reading and writing NPY and NPZ files
/// The libnpy library provides a means to read and write NPY and NPY
/// files from C++. methods for reading and writing the numpy lib (NPY) format.
/// The implementation is based upon the description available at:
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
/// The NPZ implementation draws heavily from the PKZIP Application note:
/// https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT.
///
/// Copyright (C) 2021 Matthew Johnson
///
/// For conditions of distribution and use, see copyright notice in LICENSE
///
/// ----------------------------------------------------------------------------

#ifndef _NPY_H_
#define _NPY_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define NPY_VERSION_MAJOR 2
#define NPY_VERSION_MINOR 0
#define NPY_VERSION_PATCH 0
#define NPY_VERSION_STRING "2.0.0"

const int STATIC_HEADER_LENGTH = 10;

namespace npy {
/// @brief Enumeration which represents a type of endianness
enum class endian_t : char {
  /// Indicates that the native endianness should be used. Native in this case
  /// means that of the hardware the program is currently running on.
  NATIVE,
  /// Indicates the use of big-endian encoding
  BIG,
  /// Indicates the use of little-endian encoding
  LITTLE
};

/// @brief This function will return the endianness of the current hardware.
inline endian_t native_endian() {
  union {
    std::uint32_t i;
    char c[4];
  } endian_test = {0x01020304};

  return endian_test.c[0] == 1 ? endian_t::BIG : endian_t::LITTLE;
};

/// @brief This enum represents the different types of tensor data that can be
/// stored.
enum class data_type_t : char {
  /// 8 bit signed integer
  INT8,
  /// 8 bit unsigned integer
  UINT8,
  /// 16-bit signed integer (short)
  INT16,
  /// 16-bit unsigned integer (ushort)
  UINT16,
  /// 32-bit signed integer (int)
  INT32,
  /// 32-bit unsigned integer (uint)
  UINT32,
  /// 64-bit integer (long)
  INT64,
  /// 64-bit unsigned integer (long)
  UINT64,
  /// 32-bit floating-point value (float)
  FLOAT32,
  /// 64-bit floating-point value (double)
  FLOAT64,
  /// 64-bit complex number (std::complex<float>)
  COMPLEX64,
  /// 128-bit complex number (std::complex<double>)
  COMPLEX128,
  /// Unicode string (std::wstring)
  UNICODE_STRING
};

/// @brief Convert a data type and endianness to a NPY dtype string.
/// @param dtype the data type
/// @param endian the endianness. Defaults to the current endianness of the
/// caller.
/// @return the NPY dtype string
const std::string &to_dtype(data_type_t dtype,
                            endian_t endian = endian_t::NATIVE);

/// Converts from an NPY dtype string to a data type and endianness.
/// @param dtype the NPY dtype string
/// @return a pair of data type and endianness corresponding to the input
const std::pair<data_type_t, endian_t> &from_dtype(const std::string &dtype);

std::ostream &operator<<(std::ostream &os, const endian_t &obj);
std::ostream &operator<<(std::ostream &os, const data_type_t &obj);

/// @brief Class representing the header info for an NPY file
struct header_info {
  /// Constructor.
  /// @param dictionary a Python-encoded dictionary containing the header
  /// information
  explicit header_info(const std::string &dictionary);

  /// Constructor
  header_info(data_type_t dtype, npy::endian_t endianness, bool fortran_order,
              const std::vector<size_t> &shape);

  /// The data type of the NPY file
  data_type_t dtype;

  /// The endianness of the data in the NPY file
  npy::endian_t endianness;

  /// Whether the values in the tensor are stored in FORTRAN, or column major,
  /// order
  bool fortran_order;

  /// A vector of values indicating the shape of each dimension of the tensor.
  std::vector<size_t> shape;

  /// Value used to indicate the maximum length of an element (used by Unicode
  /// strings)
  std::size_t max_element_length;
};

/// @brief Writes an NPY header to the provided stream.
/// @param output the output stream
/// @param dtype the NPY-encoded dtype string (includes data type and
/// endianness)
/// @param fortran_order whether the data is encoded in FORTRAN (i.e. column
/// major) order
/// @param shape a sequence of values indicating the shape of each dimension of
/// the tensor
/// @sa npy::to_dtype
template <typename CHAR>
void write_npy_header(std::basic_ostream<CHAR> &output,
                      const std::string &dtype, bool fortran_order,
                      const std::vector<size_t> &shape) {
  std::ostringstream buff;
  buff << "{'descr': '" << dtype;
  buff << "', 'fortran_order': " << (fortran_order ? "True" : "False");
  buff << ", 'shape': (";
  for (auto dim = shape.begin(); dim < shape.end(); ++dim) {
    buff << *dim;
    if (dim < shape.end() - 1) {
      buff << ", ";
    }
  }

  if (shape.size() == 1) {
    buff << ",";
  }

  buff << "), }";
  std::string dictionary = buff.str();
  auto dict_length = dictionary.size() + 1;
  std::string end = "\n";
  auto header_length = dict_length + STATIC_HEADER_LENGTH;
  if (header_length % 64 != 0) {
    header_length = ((header_length / 64) + 1) * 64;
    dict_length = header_length - STATIC_HEADER_LENGTH;
    end = std::string(dict_length - dictionary.length(), ' ');
    end.back() = '\n';
  }

  const char header[STATIC_HEADER_LENGTH] = {
      static_cast<char>(0x93),        'N', 'U', 'M', 'P', 'Y', 0x01, 0x00,
      static_cast<char>(dict_length), 0x00};
  output.write(header, STATIC_HEADER_LENGTH);
  output.write(reinterpret_cast<const CHAR *>(dictionary.data()),
               dictionary.length());
  output.write(reinterpret_cast<const CHAR *>(end.data()), end.length());
}

/// @brief Write values to the provided stream.
/// @tparam T the data type
/// @tparam CHAR the character type of the output stream
/// @param output the output stream
/// @param data_ptr pointer to the start of the data buffer
/// @param num_elements the number of elements to write
/// @param endianness the endianness to use in writing the data
template <typename T, typename CHAR>
void write_values(std::basic_ostream<CHAR> &output, const T *data_ptr,
                  size_t num_elements, endian_t endianness);

/// @brief Saves a tensor to the provided stream.
/// @tparam T the tensor type
/// @tparam CHAR the character type of the output stream
/// @param output the output stream
/// @param tensor the tensor
/// @param endianness the endianness to use in saving the tensor
template <typename T, typename CHAR>
void save(std::basic_ostream<CHAR> &output, const T &tensor,
          endian_t endianness = npy::endian_t::NATIVE) {
  std::vector<size_t> shape;
  for (size_t d = 0; d < tensor.ndim(); ++d) {
    shape.push_back(tensor.shape(d));
  }

  write_npy_header(output, tensor.dtype(endianness), tensor.fortran_order(),
                   shape);
  tensor.save(output, endianness);
};

/// @brief Saves a tensor to the provided stream.
/// @tparam T the data type
/// @tparam TENSOR the tensor type
/// @tparam CHAR the character type of the output stream
/// @param output the output stream
/// @param tensor the tensor
/// @param endianness the endianness to use in saving the tensor
template <typename T, template <typename> class TENSOR, typename CHAR>
void save(std::basic_ostream<CHAR> &output, const TENSOR<T> &tensor,
          endian_t endianness = npy::endian_t::NATIVE) {
  save<TENSOR<T>, CHAR>(output, tensor, endianness);
}

/// @brief Saves a tensor to the provided location on disk.
/// @tparam T the tensor type
/// @param path a path to a valid location on disk
/// @param tensor the tensor
/// @param endianness the endianness to use in saving the tensor
template <typename T>
void save(const std::string &path, T &tensor,
          endian_t endianness = npy::endian_t::NATIVE) {
  std::ofstream output(path, std::ios::out | std::ios::binary);
  if (!output.is_open()) {
    throw std::invalid_argument("path");
  }

  save<T>(output, tensor, endianness);
};

/// @brief Saves a tensor to the provided location on disk.
/// @tparam T the data type
/// @tparam TENSOR the tensor type
/// @param path a path to a valid location on disk
/// @param tensor the tensor
/// @param endianness the endianness to use in saving the tensor
template <typename T, template <typename> class TENSOR>
void save(const std::string &path, T &tensor,
          endian_t endianness = npy::endian_t::NATIVE) {
  save<TENSOR<T>>(path, tensor, endianness);
};

/// @brief Read an NPY header from the provided stream.
/// @param input the input stream
/// @return the header information
template <typename CHAR>
header_info read_npy_header(std::basic_istream<CHAR> &input) {
  std::uint8_t header[STATIC_HEADER_LENGTH];
  input.read(reinterpret_cast<CHAR *>(header), STATIC_HEADER_LENGTH);
  assert(header[0] == 0x93);
  assert(header[1] == 'N');
  assert(header[2] == 'U');
  assert(header[3] == 'M');
  assert(header[4] == 'P');
  assert(header[5] == 'Y');
  size_t dict_length = 0;
  if (header[6] == 0x01 && header[7] == 0x00) {
    dict_length = header[8] | (header[9] << 8);
  } else if (header[6] == 0x02 && header[7] == 0x00) {
    std::uint8_t extra[2];
    input.read(reinterpret_cast<CHAR *>(extra), 2);
    dict_length =
        header[8] | (header[9] << 8) | (extra[0] << 16) | (extra[1] << 24);
  }

  std::vector<CHAR> buffer(dict_length);
  input.read(buffer.data(), dict_length);
  std::string dictionary(buffer.begin(), buffer.end());
  return header_info(dictionary);
}

/// @brief Read values from the provided stream.
/// @tparam T the data type
/// @tparam CHAR the character type of the input stream
/// @param input the input stream
/// @param data_ptr pointer to the start of the data buffer
/// @param num_elements the number of elements to read
/// @param info the header information
template <typename T, typename CHAR>
void read_values(std::basic_istream<CHAR> &input, T *data_ptr,
                 size_t num_elements, const header_info &info);

template <typename T, typename CHAR> T load(std::basic_istream<CHAR> &input) {
  header_info info = read_npy_header(input);
  return T::load(input, info);
}

/// @brief Loads a tensor in NPY format from the specified location on the disk.
/// The type of the tensor must match the data to be read.
/// @tparam T the data type
/// @tparam TENSOR the tensor type
/// @param path a valid location on the disk
/// @return an object of type TENSOR<T> read from the stream
template <typename T> T load(const std::string &path) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    throw std::invalid_argument("path");
  }

  return load<T>(input);
}

/// @brief Loads a tensor in NPY format from the specified location on the disk.
/// The type of the tensor must match the data to be read.
/// @tparam T the data type
/// @tparam TENSOR the tensor type
/// @param path a valid location on the disk
/// @return an object of type TENSOR<T> read from the stream
template <typename T, template <typename> class TENSOR>
TENSOR<T> load(const std::string &path) {
  return load<TENSOR<T>>(path);
}

/// @brief Return the header information for an NPY file.
/// @param input the input stream containing the NPY-encoded bytes
/// @return the NPY header information
template <typename CHAR> header_info peek(std::basic_istream<CHAR> &input) {
  return read_npy_header(input);
}

/// @brief Return the header information for an NPY file.
/// @param path the path to the NPY file on disk
/// @return the NPY header information
header_info peek(const std::string &path);

/// @brief Enumeration indicating the compression method to use for data in the
/// NPZ archive.
enum class compression_method_t : std::uint16_t {
  /// Store the data with no compression
  STORED = 0,
  /// Use the DEFLATE algorithm to compress the data
  DEFLATED = 8
};

/// @brief Struct representing a file in the NPZ archive.
struct file_entry {
  /// The name of the file
  std::string filename;
  /// The CRC32 checksum of the uncompressed data
  std::uint32_t crc32;
  /// The size of the compressed data
  std::uint64_t compressed_size;
  /// The size of the uncompressed data
  std::uint64_t uncompressed_size;
  /// The method used to compress the data
  std::uint16_t compression_method;
  /// The offset of the file in the archive
  std::uint64_t offset;

  /// Check if this entry matches another entry
  /// @param other the other entry
  /// @return if these entries match
  bool check(const file_entry &other) const;
};

/// @brief Class which handles writing of an NPZ to an in-memory string stream.
class npzstringwriter {
public:
  /// @brief Constructor.
  /// @param compression how the entries should be compressed
  /// @param endianness the endianness to use in writing the entries
  npzstringwriter(
      compression_method_t compression = compression_method_t::STORED,
      endian_t endianness = npy::endian_t::NATIVE);

  /// @brief Destructor. This will call @ref npy::npzstringwriter::close, if it has
  /// not been called already.
  ~npzstringwriter();

  /// @brief Returns the contents of the string stream as a string.
  /// @return the state of the in-memory stream
  std::string str() const;

  /// @brief Writes the directory and end-matter of the NPZ file. Further writes
  /// will fail.
  void close();

  /// @brief Write a tensor to the NPZ archive.
  /// @tparam T the tensor type
  /// @param filename the name of the file in the archive
  /// @param tensor the tensor to write
  template <typename T>
  void write(const std::string &filename, const T &tensor) {
    if (m_closed) {
      throw std::runtime_error("Stream is closed");
    }

    std::ostringstream output;
    save<T>(output, tensor, m_endianness);

    std::string suffix = ".npy";
    std::string name = filename;
    if (name.size() < 4 ||
        !std::equal(suffix.rbegin(), suffix.rend(), name.rbegin())) {
      name += ".npy";
    }

    write_file(name, output.str());
  }

private:
  /// Write a file to the stream.
  /// @param filename the name of the file
  /// @param bytes the file data
  void write_file(const std::string &filename, std::string &&bytes);

  bool m_closed;
  std::ostringstream m_output;
  compression_method_t m_compression_method;
  endian_t m_endianness;
  std::vector<file_entry> m_entries;
};

/// @brief Class which handles writing of an NPZ archive to disk.
class npzfilewriter {
public:
  /// @brief Constructor.
  /// @param path path to the output NPZ file
  /// @param compression how the entries should be compressed
  /// @param endianness the endianness to use in writing the entries
  npzfilewriter(const std::string &path,
                compression_method_t compression = compression_method_t::STORED,
                endian_t endianness = npy::endian_t::NATIVE);

  /// @brief Constructor.
  /// @param path path to the output NPZ file
  /// @param compression how the entries should be compressed
  /// @param endianness the endianness to use in writing the entries
  npzfilewriter(const char *path,
                compression_method_t compression = compression_method_t::STORED,
                endian_t endianness = npy::endian_t::NATIVE);

  /// @brief Constructor.
  /// @param path path to the output NPZ file
  /// @param compression how the entries should be compressed
  /// @param endianness the endianness to use in writing the entries
  npzfilewriter(const std::filesystem::path &path,
                compression_method_t compression = compression_method_t::STORED,
                endian_t endianness = npy::endian_t::NATIVE);

  /// @brief Destructor. This will call @ref npy::npzfilewriter::close, if it has
  /// not been called already.
  ~npzfilewriter();

  /// @brief Returns whether the NPZ file is open.
  bool is_open() const;

  /// @brief Writes the directory and end-matter of the NPZ file, and closes the
  /// file. Further writes will fail.
  void close();

  /// @brief Write a tensor to the NPZ archive.
  /// @tparam T the tensor type
  /// @param filename the name of the file in the archive
  /// @param tensor the tensor to write
  template <typename T>
  void write(const std::string &filename, const T &tensor) {
    if (m_closed) {
      throw std::runtime_error("Stream is closed");
    }

    std::ostringstream output;
    save<T>(output, tensor, m_endianness);

    std::string suffix = ".npy";
    std::string name = filename;
    if (name.size() < 4 ||
        !std::equal(suffix.rbegin(), suffix.rend(), name.rbegin())) {
      name += ".npy";
    }

    write_file(name, output.str());
  }

private:
  /// @brief Write a file to the stream.
  /// @param filename the name of the file
  /// @param bytes the file data
  void write_file(const std::string &filename, std::string &&bytes);

  bool m_closed;
  std::ofstream m_output;
  compression_method_t m_compression_method;
  endian_t m_endianness;
  std::vector<file_entry> m_entries;
};

/// @brief Class handling reading of an NPZ from an in-memory string stream.
class npzstringreader {
public:
  /// @brief Constructor.
  /// @param bytes the contents of the stream
  npzstringreader(const std::string &bytes);

  /// @brief Constructor.
  /// @param bytes the contents of the stream
  npzstringreader(std::string &&bytes);

  /// @brief The keys of the tensors in the NPZ
  const std::vector<std::string> &keys() const;

  /// @brief Returns whether this NPZ contains the specified tensor
  /// @param filename the name of the tensor in the archive
  /// @return whether the tensor is in the archive
  bool contains(const std::string &filename);

  /// @brief Returns the header for a specified tensor.
  /// @param filename the name of the tensor in the archive
  /// @return the header for the tensor
  header_info peek(const std::string &filename);

  /// @brief Read a tensor from the archive.
  /// @details This method will throw an exception if
  /// the tensor does not exist, or if the data type of the tensor does not
  /// match the template type.
  /// @tparam T the tensor type
  /// @param filename the name of the tensor in the archive
  /// @return an instance of T read from the archive
  /// @sa npy::tensor
  template <typename T> T read(const std::string &filename) {
    std::istringstream stream(read_file(filename));
    return load<T>(stream);
  }

  template <typename T, template <typename> class TENSOR>
  TENSOR<T> read(const std::string &filename) {
    return read<TENSOR<T>>(filename);
  }

private:
  /// @brief Reads the bytes for a file from the archive.
  /// @param filename the name of the file
  /// @return the raw file bytes
  std::string read_file(const std::string &filename);

  /// @brief Read all entries from the directory.
  void read_entries();

  std::istringstream m_input;
  std::map<std::string, file_entry> m_entries;
  std::vector<std::string> m_keys;
};

/// @brief Class handling reading of an NPZ from a file on disk.
class npzfilereader {
public:
  /// @brief Constructor.
  /// @param path path to the input NPZ file
  npzfilereader(const std::string &path);

  /// @brief Constructor.
  /// @param path path to the input NPZ file
  npzfilereader(const char *path);

  /// @brief Constructor.
  /// @param path path to the input NPZ file
  npzfilereader(const std::filesystem::path &path);

  /// @brief Whether the NPZ file is open.
  bool is_open() const;

  /// @brief Closes the NPZ file.
  void close();

  /// @brief The keys of the tensors in the NPZ
  const std::vector<std::string> &keys() const;

  /// @brief Returns whether this NPZ contains the specified tensor
  /// @param filename the name of the tensor in the archive
  /// @return whether the tensor is in the archive
  bool contains(const std::string &filename);

  /// @brief Returns the header for a specified tensor.
  /// @param filename the name of the tensor in the archive
  /// @return the header for the tensor
  header_info peek(const std::string &filename);

  /// @brief Read a tensor from the archive.
  /// @details This method will throw an exception if
  /// the tensor does not exist, or if the data type of the tensor does not
  /// match the template type.
  /// @tparam T the tensor type
  /// @param filename the name of the tensor in the archive
  /// @return an instance of T read from the archive
  template <typename T> T read(const std::string &filename) {
    std::istringstream stream(read_file(filename));
    return load<T>(stream);
  }

  /// @brief Read a tensor from the archive.
  /// @details This method will throw an exception if
  /// the tensor does not exist, or if the data type of the tensor does not
  /// match the template type.
  /// @tparam T the data type
  /// @tparam TENSOR the tensor type
  /// @param filename the name of the tensor in the archive
  /// @return an instance of TENSOR<T> read from the archive
  template <typename T, template <typename> class TENSOR>
  TENSOR<T> read(const std::string &filename) {
    read<TENSOR<T>>(filename);
  }

private:
  /// @brief Reads the bytes for a file from the archive.
  /// @param filename the name of the file
  /// @return the raw file bytes
  std::string read_file(const std::string &filename);

  /// @brief Read all entries from the directory.
  void read_entries();

  std::ifstream m_input;
  std::map<std::string, file_entry> m_entries;
  std::vector<std::string> m_keys;
};

/// @brief The default tensor class.
/// @details This class can be used as a data exchange format
/// for the library, but the methods and classes will also work with your own
/// tensor implementation. The library methods require the following methods to
/// be present in a tensor type:
/// - @ref load
/// - @ref save
/// - @ref shape
/// - @ref ndim
/// - @ref dtype
/// - @ref fortran_order
///
/// As long as these are present and have the same semantics, the library
/// should handle them in the same was as this implementation. Only certain type
/// of tensor objects are natively supported (see @ref npy::data_type_t).
/// @note This class is not optimized for access speed. It is intended as a
/// simple data exchange format. Once the raw data has been extracted from the
/// NPY or NPZ, it is recommended to convert it to a more efficient format for
/// processing using the data() method.
template <typename T> class tensor {
public:
  /// The value type of the tensor.
  typedef T value_type;
  /// The reference type of the tensor.
  typedef value_type &reference;
  /// The const reference type of the tensor.
  typedef const value_type &const_reference;
  /// The pointer type of the tensor.
  typedef value_type *pointer;
  /// The const pointer type of the tensor.
  typedef const value_type *const_pointer;

  /// @brief Constructor.
  /// @details This will allocate a data buffer of the appropriate size in
  /// row-major order.
  /// @param shape the shape of the tensor
  tensor(const std::vector<size_t> &shape) : tensor(shape, false) {}

  /// @brief Constructor.
  /// @details This will allocate a data buffer of the appropriate size.
  /// @param shape the shape of the tensor
  /// @param fortran_order whether the data is stored in FORTRAN, or column
  /// major, order
  tensor(const std::vector<size_t> &shape, bool fortran_order)
      : m_shape(shape),
        m_ravel_strides(tensor<T>::get_ravel_strides(shape, fortran_order)),
        m_fortran_order(fortran_order), m_dtype(tensor<T>::get_dtype()),
        m_values(tensor<T>::get_size(shape)) {}

  /// @brief Copy constructor.
  tensor(const tensor<T> &other)
      : m_shape(other.m_shape), m_ravel_strides(other.m_ravel_strides),
        m_fortran_order(other.m_fortran_order), m_dtype(other.m_dtype),
        m_values(other.m_values) {}

  /// @brief Move constructor.
  tensor(tensor<T> &&other)
      : m_shape(std::move(other.m_shape)),
        m_ravel_strides(std::move(other.m_ravel_strides)),
        m_fortran_order(other.m_fortran_order), m_dtype(other.m_dtype),
        m_values(std::move(other.m_values)) {}

  /// @brief Load a tensor from the specified location on disk.
  static tensor<T> from_file(const std::string &path) {
    return npy::load<tensor<T>>(path);
  }

  /// @brief Load a tensor from the provided stream.
  /// @details This is one of the methods required by the library to
  /// read NPY files. If you implement this in a custom tensor, you will
  /// need to populate your internal data structure using the provided
  /// stream and header information. The @ref npy::read_values method can be
  /// used to read raw data from the stream.
  /// @param input the input stream
  /// @param info the header information
  /// @return an instance of the tensor read from the stream
  /// @sa npy::read_values
  static tensor<T> load(std::basic_istream<char> &input,
                        const header_info &info) {
    tensor<T> result(info.shape, info.fortran_order);
    if (info.dtype != result.dtype()) {
      throw std::runtime_error("requested dtype does not match stream's dtype");
    }

    read_values(input, result.m_values.data(), result.m_values.size(), info);

    return result;
  }

  /// @brief Save the tensor to the provided stream.
  /// @details This is one of the methods required by the library to
  /// write NPY files. If you implement this in a custom tensor, you will
  /// need to write your internal data structure to the provided stream. The
  /// @ref npy::write_values method can be used to write raw data to the stream.
  /// @param output the output stream
  /// @param endianness the endianness to use in writing the data
  /// @sa npy::write_values
  void save(std::basic_ostream<char> &output, endian_t endianness) const {
    write_values(output, m_values.data(), m_values.size(), endianness);
  }

  /// @brief Variable parameter index function.
  /// @param index an index into the tensor. Can be negative (in which case it
  /// will work as in numpy)
  /// @return the value at the provided index
  template <typename... Indices> const T &operator()(Indices... index) const {
    return m_values[ravel(std::vector<std::int32_t>({index...}))];
  }

  /// @brief Index function.
  /// @param multi_index the index into the tensor
  /// @return the value at the provided index
  const T &operator()(const std::vector<std::size_t> &multi_index) const {
    return m_values[ravel(multi_index)];
  }

  /// @brief Variable parameter index function.
  /// @param index an index into the tensor. Can be negative (in which case it
  /// will work as in numpy)
  /// @return the value at the provided index
  template <typename... Indices> T &operator()(Indices... index) {
    return m_values[ravel(std::vector<std::int32_t>({index...}))];
  }

  /// @brief Index function.
  /// @param multi_index the index into the tensor
  /// @return the value at the provided index
  T &operator()(const std::vector<std::size_t> &multi_index) {
    return m_values[ravel(multi_index)];
  }

  /// @brief Iterator pointing at the beginning of the tensor in memory.
  typename std::vector<T>::iterator begin() { return m_values.begin(); }

  /// @brief Iterator pointing at the beginning of the tensor in memory.
  typename std::vector<T>::const_iterator begin() const {
    return m_values.begin();
  }

  /// @brief Iterator pointing at the end of the tensor in memory.
  typename std::vector<T>::iterator end() { return m_values.end(); }

  /// @brief Iterator pointing at the end of the tensor in memory.
  typename std::vector<T>::const_iterator end() const { return m_values.end(); }

  /// @brief Sets the value at the provided index.
  /// @param multi_index an index into the tensor
  /// @param value the value to set
  void set(const std::vector<std::int32_t> &multi_index, const T &value) {
    m_values[ravel(multi_index)] = value;
  }

  /// @brief Gets the value at the provided index.
  /// @param multi_index the index into the tensor
  /// @return the value at the provided index
  const T &get(const std::vector<std::int32_t> &multi_index) const {
    return m_values[ravel(multi_index)];
  }

  /// @brief The data type of the tensor.
  std::string dtype(endian_t endianness) const {
    return to_dtype(m_dtype, endianness);
  }

  /// @brief The data type of the tensor.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @return the data type of the tensor
  data_type_t dtype() const { return m_dtype; };

  /// @brief The underlying values buffer.
  const std::vector<T> &values() const { return m_values; }

  /// @brief Copy values from the source to this tensor.
  /// @param source pointer to the start of the source buffer
  /// @param nitems the number of items to copy. Should be equal to @ref size.
  void copy_from(const T *source, size_t nitems) {
    if (nitems != size()) {
      throw std::invalid_argument("nitems");
    }

    std::copy(source, source + nitems, m_values.begin());
  }

  /// @brief Copy values from the provided vector.
  /// @param source the source vector. Should have the same size as @ref values.
  void copy_from(const std::vector<T> &source) {
    if (source.size() != size()) {
      throw std::invalid_argument("source.size");
    }

    std::copy(source.begin(), source.end(), m_values.begin());
  }

  /// @brief Move values from the provided vector.
  /// @param source the source vector. Should have the same size as @ref values.
  void move_from(std::vector<T> &&source) {
    if (source.size() != size()) {
      throw std::invalid_argument("source.size");
    }

    m_values = std::move(source);
  }

  /// @brief A pointer to the start of the underlying values buffer.
  T *data() { return m_values.data(); }

  /// @brief A pointer to the start of the underlying values buffer.
  const T *data() const { return m_values.data(); }

  /// @brief The number of elements in the tensor.
  size_t size() const { return m_values.size(); }

  /// @brief The shape of the vector. Each element is the size of the
  /// corresponding dimension.
  const std::vector<size_t> &shape() const { return m_shape; }

  /// @brief Returns the dimensionality of the tensor at the specified index.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @param index index into the shape
  /// @return the dimensionality at the index
  size_t shape(int index) const { return m_shape[index]; }

  /// @brief The number of dimensions of the tensor.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @return the number of dimensions
  size_t ndim() const { return m_shape.size(); }

  /// @brief Whether the tensor data is stored in FORTRAN, or column-major,
  /// order.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @return whether the tensor is stored in FORTRAN order
  bool fortran_order() const { return m_fortran_order; }

  /// @brief Copy assignment operator.
  tensor<T> &operator=(const tensor<T> &other) {
    m_shape = other.m_shape;
    m_ravel_strides = other.m_ravel_strides;
    m_fortran_order = other.m_fortran_order;
    m_dtype = other.m_dtype;
    m_values = other.m_values;
    return *this;
  }

  /// @brief Move assignment operator.
  tensor<T> &operator=(tensor<T> &&other) {
    m_shape = std::move(other.m_shape);
    m_ravel_strides = std::move(other.m_ravel_strides);
    m_fortran_order = other.m_fortran_order;
    m_dtype = other.m_dtype;
    m_values = std::move(other.m_values);
    return *this;
  }

  /// @brief Save this tensor to the provided location on disk.
  /// @param path a valid location on disk
  /// @param endianness the endianness to use in writing the tensor
  void save(const std::string &path,
            endian_t endianness = npy::endian_t::NATIVE) {
    npy::save(path, *this, endianness);
  }

  /// @brief Ravels a multi-index into a single value indexing the buffer.
  /// @tparam INDEX_IT the index iterator class
  /// @tparam SHAPE_IT the shape iterator class
  /// @param index the multi-index iterator
  /// @param shape the shape iterator
  /// @return the single value in the buffer corresponding to the multi-index
  template <class INDEX_IT, class SHAPE_IT>
  size_t ravel(INDEX_IT index, SHAPE_IT shape) const {
    std::size_t ravel = 0;
    for (auto stride = m_ravel_strides.begin(); stride < m_ravel_strides.end();
         ++index, ++shape, ++stride) {
      if (*index >= *shape) {
        throw std::invalid_argument("multi_index");
      }

      ravel += *index * *stride;
    }

    return ravel;
  }

  /// @brief Ravels a multi-index into a single value indexing the buffer.
  /// @param multi_index the multi-index value
  /// @return the single value in the buffer corresponding to the multi-index
  size_t ravel(const std::vector<std::int32_t> &multi_index) const {
    if (multi_index.size() != m_shape.size()) {
      throw std::invalid_argument("multi_index");
    }

    std::vector<std::size_t> abs_multi_index(multi_index.size());
    std::transform(multi_index.begin(), multi_index.end(), m_shape.begin(),
                   abs_multi_index.begin(),
                   [](std::int32_t index, std::size_t shape) -> std::size_t {
                     if (index < 0) {
                       return static_cast<std::size_t>(shape + index);
                     }

                     return static_cast<std::size_t>(index);
                   });

    return ravel(abs_multi_index);
  }

  /// @brief Ravels a multi-index into a single value indexing the buffer.
  /// @param abs_multi_index the multi-index value
  /// @return the single value in the buffer corresponding to the multi-index
  size_t ravel(const std::vector<std::size_t> &abs_multi_index) const {
    if (m_fortran_order) {
      return ravel(abs_multi_index.rbegin(), m_shape.rbegin());
    }

    return ravel(abs_multi_index.begin(), m_shape.begin());
  }

private:
  std::vector<size_t> m_shape;
  std::vector<size_t> m_ravel_strides;
  bool m_fortran_order;
  data_type_t m_dtype;
  std::vector<T> m_values;

  /// @brief Returns the data type for this tensor.
  static data_type_t get_dtype();

  /// @brief Gets the size of a tensor given its shape
  static size_t get_size(const std::vector<size_t> &shape) {
    size_t size = 1;
    for (auto &dim : shape) {
      size *= dim;
    }

    return size;
  }

  /// @brief Gets the strides for ravelling
  static std::vector<size_t> get_ravel_strides(const std::vector<size_t> &shape,
                                               bool fortran_order) {
    std::vector<size_t> ravel_strides(shape.size());
    size_t stride = 1;
    auto ravel = ravel_strides.rbegin();
    if (fortran_order) {
      for (auto max_index = shape.begin(); max_index < shape.end();
           ++max_index, ++ravel) {
        *ravel = stride;
        stride *= *max_index;
      }
    } else {
      for (auto max_index = shape.rbegin(); max_index < shape.rend();
           ++max_index, ++ravel) {
        *ravel = stride;
        stride *= *max_index;
      }
    }

    return ravel_strides;
  }
};

/// @brief Specialization of dtype for std::wstring tensors.
template <>
inline std::string tensor<std::wstring>::dtype(endian_t endianness) const {
  std::size_t max_length = 0;
  for (const auto &element : m_values) {
    if (element.size() > max_length) {
      max_length = element.size();
    }
  }

  if (endianness == npy::endian_t::NATIVE) {
    endianness = native_endian();
  }

  if (endianness == npy::endian_t::LITTLE) {
    return "<U" + std::to_string(max_length);
  }

  return ">U" + std::to_string(max_length);
}

} // namespace npy

#endif