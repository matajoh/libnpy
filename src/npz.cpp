#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "npy/npy.h"
#include "zip.h"

namespace {
using namespace npy;

const std::array<std::uint8_t, 4> LOCAL_HEADER_SIG = {0x50, 0x4B, 0x03, 0x04};
const std::array<std::uint8_t, 4> CD_HEADER_SIG = {0x50, 0x4B, 0x01, 0x02};
const std::array<std::uint8_t, 4> CD_END_SIG = {0x50, 0x4B, 0x05, 0x06};

const std::array<std::uint8_t, 4> EXTERNAL_ATTR = {0x00, 0x00, 0x80, 0x01};
const std::array<std::uint8_t, 4> TIME = {0x00, 0x00, 0x21, 0x00};
const int CD_END_SIZE = 22;
const std::uint16_t STANDARD_VERSION =
    20; // 2.0 File is encrypted using traditional PKWARE encryption
const std::uint16_t ZIP64_VERSION = 45; // 4.5 File uses ZIP64 format extensions

const std::uint16_t ZIP64_TAG = 1;
const std::uint64_t ZIP64_LIMIT = 0x8FFFFFFF;
const std::uint32_t ZIP64_PLACEHOLDER = 0xFFFFFFFF;

void write(std::ostream &stream, std::uint16_t value) {
  stream.put(value & 0x00FF);
  stream.put(value >> 8);
}

void write(std::ostream &stream, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    stream.put(value & 0x000000FF);
    value >>= 8;
  }
}

void write32(std::ostream &stream, std::uint64_t value) {
  if (value > ZIP64_LIMIT) {
    write(stream, ZIP64_PLACEHOLDER);
  } else {
    write(stream, static_cast<std::uint32_t>(value));
  }
}

void write(std::ostream &stream, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    stream.put(value & 0x00000000000000FF);
    value >>= 8;
  }
}

std::uint16_t read16(std::istream &stream) {
  std::uint16_t low = static_cast<std::uint16_t>(stream.get());
  std::uint16_t high = static_cast<std::uint16_t>(stream.get());
  return low | (high << 8);
}

std::uint32_t read32(std::istream &stream) {
  std::uint32_t result = 0;
  int shift = 0;
  for (int i = 0; i < 4; ++i, shift += 8) {
    std::uint32_t part = stream.get();
    result |= part << shift;
  }

  return result;
}

std::uint64_t read64(std::istream &stream) {
  std::uint64_t result = 0;
  int shift = 0;
  for (int i = 0; i < 8; ++i, shift += 8) {
    std::uint64_t part = stream.get();
    result |= part << shift;
  }

  return result;
}

void assert_sig(std::istream &stream,
                const std::array<std::uint8_t, 4> &expected,
                const char *entity) {
  std::array<std::uint8_t, 4> actual;
  stream.read(reinterpret_cast<char *>(actual.data()), actual.size());
  if (actual != expected) {
    printf("Invalid signature when reading %s:\n", entity);
    printf("actual: [%d, %d, %d, %d]\n", actual[0], actual[1], actual[2],
           actual[3]);
    printf("expected: [%d, %d, %d, %d]\n", expected[0], expected[1],
           expected[2], expected[3]);
    throw std::runtime_error("Invalid signature (Not a valid NPZ file)");
  }
}

std::uint16_t determine_extra_length(const npy::file_entry &header,
                                     bool include_offset) {
  std::uint16_t length = 0;
  if (header.compressed_size > ZIP64_LIMIT) {
    length += 8;
  }

  if (header.uncompressed_size > ZIP64_LIMIT) {
    length += 8;
  }

  if (include_offset && header.offset > ZIP64_LIMIT) {
    length += 8;
  }

  return length;
}

void write_zip64_extra(std::ostream &stream, const npy::file_entry &header,
                       bool include_offset) {
  std::vector<std::uint64_t> extra;
  if (header.uncompressed_size > ZIP64_LIMIT) {
    extra.push_back(header.uncompressed_size);
  }

  if (header.compressed_size > ZIP64_LIMIT) {
    extra.push_back(header.compressed_size);
  }

  if (include_offset && header.offset > ZIP64_LIMIT) {
    extra.push_back(header.offset);
  }

  write(stream, ZIP64_TAG);
  write(stream, static_cast<std::uint16_t>(extra.size() * 8));
  for (auto val : extra) {
    write(stream, val);
  }
}

void read_zip64_extra(std::istream &stream, npy::file_entry &header,
                      bool include_offset) {
  std::uint16_t tag = read16(stream);
  if (tag != ZIP64_TAG) {
    throw std::runtime_error("Invalid tag (expected ZIP64)");
  }

  std::uint16_t actual_size = read16(stream);
  std::uint16_t expected_size = 0;

  if (header.uncompressed_size == ZIP64_PLACEHOLDER) {
    header.uncompressed_size = read64(stream);
    expected_size += 8;
  }

  if (header.compressed_size == ZIP64_PLACEHOLDER) {
    header.compressed_size = read64(stream);
    expected_size += 8;
  }

  if (include_offset && header.offset == ZIP64_PLACEHOLDER) {
    header.offset = read64(stream);
    expected_size += 8;
  }

  if (actual_size < expected_size) {
    throw std::runtime_error("ZIP64 extra info missing");
  }

  if (actual_size > expected_size) {
    // this can be the result of force_zip64 being set in Python's zipfile
    stream.seekg(actual_size - expected_size, std::ios::cur);
  }
}

void write_shared_header(std::ostream &stream, const npy::file_entry &header) {
  std::uint16_t general_purpose_big_flag = 0;
  write(stream, general_purpose_big_flag);
  write(stream, header.compression_method);
  stream.write(reinterpret_cast<const char *>(TIME.data()), TIME.size());
  write(stream, header.crc32);
  write32(stream, header.compressed_size);
  write32(stream, header.uncompressed_size);
  write(stream, static_cast<std::uint16_t>(header.filename.length()));
}

std::uint16_t read_shared_header(std::istream &stream,
                                 npy::file_entry &header) {
  read16(stream); // general purpose bit flag
  header.compression_method = read16(stream);
  read32(stream); // time
  header.crc32 = read32(stream);
  header.compressed_size = read32(stream);
  header.uncompressed_size = read32(stream);
  return read16(stream);
}

void write_local_header(std::ostream &stream, const npy::file_entry &header,
                        bool zip64) {
  stream.write(reinterpret_cast<const char *>(LOCAL_HEADER_SIG.data()),
               LOCAL_HEADER_SIG.size());
  write(stream, zip64 ? ZIP64_VERSION : STANDARD_VERSION);
  write_shared_header(stream, header);
  std::uint16_t extra_field_length = determine_extra_length(header, false);
  write(stream, extra_field_length);
  stream.write(header.filename.data(), header.filename.length());
  if (extra_field_length > 0) {
    write_zip64_extra(stream, header, false);
  }
}

npy::file_entry read_local_header(std::istream &stream) {
  assert_sig(stream, LOCAL_HEADER_SIG, "local_header");
  std::uint16_t version = read16(stream);
  if (version > ZIP64_VERSION) {
    throw std::runtime_error("Unsupported NPZ version");
  }

  npy::file_entry entry;

  std::uint16_t filename_length = read_shared_header(stream, entry);
  std::uint16_t extra_field_length = read16(stream);
  std::vector<char> buffer(filename_length);
  stream.read(buffer.data(), filename_length);
  entry.filename = std::string(buffer.begin(), buffer.end());

  if (extra_field_length > 0) {
    read_zip64_extra(stream, entry, false);
  }

  return entry;
}

void write_central_directory_header(std::ostream &stream,
                                    const npy::file_entry &header) {
  std::uint16_t extra_field_length = determine_extra_length(header, true);
  stream.write(reinterpret_cast<const char *>(CD_HEADER_SIG.data()),
               CD_HEADER_SIG.size());
  write(stream, STANDARD_VERSION);
  write(stream, extra_field_length > 0 ? ZIP64_VERSION : STANDARD_VERSION);
  write_shared_header(stream, header);
  write(stream, extra_field_length);
  std::uint16_t file_comment_length = 0;
  write(stream, file_comment_length);
  std::uint16_t disk_number_start = 0;
  write(stream, disk_number_start);
  std::uint16_t internal_file_attributes = 0;
  write(stream, internal_file_attributes);
  stream.write(reinterpret_cast<const char *>(EXTERNAL_ATTR.data()),
               EXTERNAL_ATTR.size());
  write32(stream, header.offset);
  stream.write(header.filename.data(), header.filename.length());
  if (extra_field_length > 0) {
    write_zip64_extra(stream, header, true);
  }
}

npy::file_entry read_central_directory_header(std::istream &stream) {
  assert_sig(stream, CD_HEADER_SIG, "central_directory");
  read16(stream); // version made by
  std::uint16_t version = read16(stream);
  if (version > ZIP64_VERSION) {
    throw std::runtime_error("Unsupported NPZ version");
  }

  npy::file_entry entry;
  std::uint16_t filename_length = read_shared_header(stream, entry);
  std::uint16_t extra_field_length = read16(stream);
  read16(stream); // file comment length
  read16(stream); // disk number start
  read16(stream); // internal file attributes
  read32(stream); // external file attributes
  entry.offset = read32(stream);

  std::vector<char> buffer(filename_length);
  stream.read(buffer.data(), filename_length);
  entry.filename = std::string(buffer.begin(), buffer.end());

  if (extra_field_length > 0) {
    read_zip64_extra(stream, entry, true);
  }

  return entry;
}

struct CentralDirectory {
  std::uint16_t num_entries;
  std::uint32_t size;
  std::uint32_t offset;
};

void write_end_of_central_directory(std::ostream &stream,
                                    const CentralDirectory &dir) {
  stream.write(reinterpret_cast<const char *>(CD_END_SIG.data()),
               CD_END_SIG.size());
  uint16_t disk_number = 0;
  write(stream, disk_number);
  write(stream, disk_number);
  write(stream, dir.num_entries);
  write(stream, dir.num_entries);
  write(stream, dir.size);
  write(stream, dir.offset);
  std::uint16_t file_comment_length = 0;
  write(stream, file_comment_length);
}

CentralDirectory read_end_of_central_directory(std::istream &stream) {
  assert_sig(stream, CD_END_SIG, "end_of_central_directory");

  CentralDirectory result;
  read16(stream); // number of this disk
  read16(stream); // number of the disk with the start of the central directory
  result.num_entries = read16(stream);
  read16(stream); // num_entries_on_disk
  result.size = read32(stream);
  result.offset = read32(stream);
  return result;
}

void read_entries(std::istream &input,
                  std::map<std::string, file_entry> &entries,
                  std::vector<std::string> &keys) {
  input.seekg(-CD_END_SIZE, std::ios::end);
  CentralDirectory dir = read_end_of_central_directory(input);

  input.seekg(dir.offset, std::ios::beg);

  for (size_t i = 0; i < dir.num_entries; ++i) {
    file_entry entry = read_central_directory_header(input);
    entries[entry.filename] = entry;
    keys.push_back(entry.filename);
  }

  std::sort(keys.begin(), keys.end());
}

void close(std::ostream &output, const std::vector<file_entry> &entries) {
  CentralDirectory dir;
  dir.offset = static_cast<std::uint32_t>(output.tellp());
  for (auto &header : entries) {
    write_central_directory_header(output, header);
  }

  dir.size = static_cast<std::uint32_t>(output.tellp()) - dir.offset;
  dir.num_entries = static_cast<std::uint16_t>(entries.size());
  write_end_of_central_directory(output, dir);
}

void write_file(std::ostream &output, std::vector<file_entry> &entries,
                const std::string &filename,
                compression_method_t compression_method, std::string &&bytes) {
  std::uint32_t uncompressed_size = static_cast<std::uint32_t>(bytes.size());
  std::uint32_t compressed_size = 0;
  std::string compressed_bytes;
  std::uint32_t checksum = npy_crc32(bytes);
  if (compression_method == compression_method_t::STORED) {
    compressed_bytes = bytes;
    compressed_size = uncompressed_size;
  } else if (compression_method == compression_method_t::DEFLATED) {
    compressed_bytes = npy_deflate(std::move(bytes));
    compressed_size = static_cast<std::uint32_t>(compressed_bytes.size());
  } else {
    throw std::invalid_argument("Unsupported compression method");
  }

  file_entry entry = {filename,
                      checksum,
                      compressed_size,
                      uncompressed_size,
                      static_cast<std::uint16_t>(compression_method),
                      static_cast<std::uint32_t>(output.tellp())};

  bool zip64 = uncompressed_size > ZIP64_LIMIT || compressed_size > ZIP64_LIMIT;
  write_local_header(output, entry, zip64);
  output.write(compressed_bytes.data(), compressed_size);
  entries.push_back(std::move(entry));
}

typedef union crcbytes_u {
  std::uint32_t value;
  std::uint8_t bytes[4];
} CRCBytes;

std::string read_file(std::istream &input,
                      const std::map<std::string, file_entry> &entries,
                      const std::string &temp_filename) {
  std::string filename = temp_filename;
  if (entries.count(filename) == 0) {
    filename += ".npy";
    if (entries.count(filename) == 0) {
      throw std::invalid_argument("filename");
    }
  }

  const file_entry &entry = entries.at(filename);
  input.seekg(entry.offset, std::ios::beg);

  file_entry local = read_local_header(input);
  if (!entry.check(local)) {
    throw std::runtime_error("Central directory and local headers disagree");
  }

  std::string uncompressed_bytes;
  uncompressed_bytes.resize(entry.compressed_size);
  input.read(reinterpret_cast<char *>(uncompressed_bytes.data()),
             uncompressed_bytes.size());
  compression_method_t cmethod =
      static_cast<compression_method_t>(entry.compression_method);
  if (cmethod == compression_method_t::DEFLATED) {
    uncompressed_bytes = npy_inflate(std::move(uncompressed_bytes));
  }

  std::uint32_t actual_crc32 = npy_crc32(uncompressed_bytes);
  if (actual_crc32 != entry.crc32) {
    CRCBytes actual_bytes{actual_crc32};
    CRCBytes expected_bytes{entry.crc32};
    printf("CRC mismatch when reading %s:\n", filename.c_str());
    printf("actual: [0x%x, 0x%x, 0x%x, 0x%x]\n", actual_bytes.bytes[0],
           actual_bytes.bytes[1], actual_bytes.bytes[2], actual_bytes.bytes[3]);
    printf("expected: [0x%x, 0x%x, 0x%x, 0x%x]\n", expected_bytes.bytes[0],
           expected_bytes.bytes[1], expected_bytes.bytes[2],
           expected_bytes.bytes[3]);
    throw std::runtime_error("CRC mismatch");
  }

  return uncompressed_bytes;
}

} // namespace

namespace npy {
bool file_entry::check(const file_entry &other) const {
  return !(other.filename != this->filename || other.crc32 != this->crc32 ||
           other.compression_method != this->compression_method ||
           other.compressed_size != this->compressed_size ||
           other.uncompressed_size != this->uncompressed_size);
}

npzstringwriter::npzstringwriter(compression_method_t compression,
                                 endian_t endianness)
    : m_closed(false), m_compression_method(compression),
      m_endianness(endianness) {}

npzstringwriter::~npzstringwriter() {
  if (!m_closed) {
    close();
  }
}

std::string npzstringwriter::str() const { return m_output.str(); }

void npzstringwriter::write_file(const std::string &filename,
                                 std::string &&bytes) {
  if (m_closed) {
    throw std::runtime_error("NPZ file has been closed");
  }

  ::write_file(m_output, m_entries, filename, m_compression_method,
               std::move(bytes));
}

void npzstringwriter::close() {
  if (!m_closed) {
    ::close(m_output, m_entries);
    m_closed = true;
  }
}

npzfilewriter::npzfilewriter(const std::string &path,
                             compression_method_t compression,
                             endian_t endianness)
    : m_closed(false), m_output(path, std::ios::binary),
      m_compression_method(compression), m_endianness(endianness) {}

npzfilewriter::npzfilewriter(const std::filesystem::path &path,
                             compression_method_t compression,
                             endian_t endianness)
    : m_closed(false), m_output(path, std::ios::binary),
      m_compression_method(compression), m_endianness(endianness) {}

npzfilewriter::npzfilewriter(const char *path, compression_method_t compression,
                             endian_t endianness)
    : m_closed(false), m_output(path, std::ios::binary),
      m_compression_method(compression), m_endianness(endianness) {}

npzfilewriter::~npzfilewriter() {
  if (!m_closed) {
    close();
  }
}

bool npzfilewriter::is_open() const { return m_output.is_open(); }

void npzfilewriter::write_file(const std::string &filename,
                               std::string &&bytes) {
  if (m_closed) {
    throw std::runtime_error("NPZ file has been closed");
  }

  ::write_file(m_output, m_entries, filename, m_compression_method,
               std::move(bytes));
}

void npzfilewriter::close() {
  if (!m_closed) {
    ::close(m_output, m_entries);
    m_closed = true;
    m_output.close();
  }
}

npzstringreader::npzstringreader(const std::string &bytes) : m_input(bytes) {
  read_entries();
}

npzstringreader::npzstringreader(std::string &&bytes)
    : m_input(std::move(bytes)) {
  read_entries();
}

void npzstringreader::read_entries() {
  ::read_entries(m_input, m_entries, m_keys);
}

const std::vector<std::string> &npzstringreader::keys() const { return m_keys; }

std::string npzstringreader::read_file(const std::string &filename) {
  return ::read_file(m_input, m_entries, filename);
}

bool npzstringreader::contains(const std::string &filename) {
  return m_entries.count(filename);
}

header_info npzstringreader::peek(const std::string &filename) {
  std::istringstream stream(read_file(filename));
  return npy::peek(stream);
}

npzfilereader::npzfilereader(const std::string &path)
    : m_input(path, std::ios::binary) {
  read_entries();
}

npzfilereader::npzfilereader(const char *path)
    : m_input(path, std::ios::binary) {
  read_entries();
}

npzfilereader::npzfilereader(const std::filesystem::path &path)
    : m_input(path, std::ios::binary) {
  read_entries();
}

void npzfilereader::read_entries() {
  if (!m_input.is_open()) {
    throw std::invalid_argument("File not found");
  }

  ::read_entries(m_input, m_entries, m_keys);
}

const std::vector<std::string> &npzfilereader::keys() const { return m_keys; }

std::string npzfilereader::read_file(const std::string &filename) {
  return ::read_file(m_input, m_entries, filename);
}

bool npzfilereader::contains(const std::string &filename) {
  return m_entries.count(filename);
}

header_info npzfilereader::peek(const std::string &filename) {
  std::istringstream stream(read_file(filename));
  return npy::peek(stream);
}

bool npzfilereader::is_open() const { return m_input.is_open(); }

void npzfilereader::close() { m_input.close(); }

} // namespace npy