#include <string>
#include <sstream>
#include <cstdint>
#include <array>
#include <memory>
#include <iostream>
#include <cassert>

#include "npz.h"
#include "zip.h"

namespace
{
const std::array<std::uint8_t, 4> LOCAL_HEADER_SIG = {0x50, 0x4B, 0x03, 0x04};
const std::array<std::uint8_t, 4> CD_HEADER_SIG = {0x50, 0x4B, 0x01, 0x02};
const std::array<std::uint8_t, 4> CD_END_SIG = {0x50, 0x4B, 0x05, 0x06};

const std::array<std::uint8_t, 4> EXTERNAL_ATTR = {0x00, 0x00, 0x80, 0x01};
const std::array<std::uint8_t, 4> TIME = {0x00, 0x00, 0x21, 0x00};
const int CD_END_SIZE = 22;
const std::uint16_t VERSION = 20;
const int CHUNK = 128 * 1024;

void write(std::ostream &stream, std::uint16_t value)
{
    stream.put(value & 0x00FF);
    stream.put(value >> 8);
}

void write(std::ostream &stream, std::uint32_t value)
{
    for (int i = 0; i < 4; ++i)
    {
        stream.put(value & 0x000000FF);
        value >>= 8;
    }
}

std::uint16_t read16(std::istream &stream)
{
    std::uint8_t low = stream.get();
    std::uint8_t high = stream.get();
    return low | (high << 8);
}

std::uint32_t read32(std::istream &stream)
{
    std::uint32_t result = 0;
    int shift = 0;
    for (int i = 0; i < 4; ++i, shift += 8)
    {
        std::uint8_t part = stream.get();
        result |= part << shift;
    }

    return result;
}

void assert_sig(std::istream &stream, const std::array<std::uint8_t, 4> &expected)
{
    std::array<std::uint8_t, 4> actual;
    stream.read(reinterpret_cast<char *>(actual.data()), actual.size());
    if (actual != expected)
    {
        throw std::logic_error("Invalid signature (Not a valid NPZ file)");
    }
}

void write_shared_header(std::ostream &stream, const npy::file_entry &header)
{
    std::uint16_t general_purpose_big_flag = 0;
    write(stream, general_purpose_big_flag);
    write(stream, header.compression_method);
    stream.write(reinterpret_cast<const char *>(TIME.data()), TIME.size());
    write(stream, header.crc32);
    write(stream, header.compressed_size);
    write(stream, header.uncompressed_size);
    write(stream, static_cast<std::uint16_t>(header.filename.length()));
    std::uint16_t extra_field_length = 0;
    write(stream, extra_field_length);
}

std::size_t read_shared_header(std::istream &stream, npy::file_entry &header)
{
    read16(stream); // general purpose bit flag
    header.compression_method = read16(stream);
    read32(stream); // time
    header.crc32 = read32(stream);
    header.compressed_size = read32(stream);
    header.uncompressed_size = read32(stream);
    size_t length = read16(stream);
    read16(stream); // extra field length
    return length;
}

void write_local_header(std::ostream &stream, const npy::file_entry &header)
{
    stream.write(reinterpret_cast<const char *>(LOCAL_HEADER_SIG.data()), LOCAL_HEADER_SIG.size());
    write(stream, VERSION);
    write_shared_header(stream, header);
    stream.write(header.filename.data(), header.filename.length());
}

npy::file_entry read_local_header(std::istream &stream)
{
    assert_sig(stream, LOCAL_HEADER_SIG);
    std::uint16_t version = read16(stream);
    if (version > VERSION)
    {
        throw std::logic_error("Unsupported NPZ version");
    }

    npy::file_entry entry;
    size_t length = read_shared_header(stream, entry);
    std::vector<char> buffer(length);
    stream.read(buffer.data(), length);
    entry.filename = std::string(buffer.begin(), buffer.end());
    return entry;
}

void write_central_directory_header(std::ostream &stream, const npy::file_entry &header)
{
    stream.write(reinterpret_cast<const char *>(CD_HEADER_SIG.data()), CD_HEADER_SIG.size());
    write(stream, VERSION);
    write(stream, VERSION);
    write_shared_header(stream, header);
    std::uint16_t file_comment_length = 0;
    write(stream, file_comment_length);
    std::uint16_t disk_number_start = 0;
    write(stream, disk_number_start);
    std::uint16_t internal_file_attributes = 0;
    write(stream, internal_file_attributes);
    stream.write(reinterpret_cast<const char *>(EXTERNAL_ATTR.data()), EXTERNAL_ATTR.size());
    write(stream, header.offset);
    stream.write(header.filename.data(), header.filename.length());
}

npy::file_entry read_central_directory_header(std::istream &stream)
{
    assert_sig(stream, CD_HEADER_SIG);
    read16(stream); // version made by
    std::uint16_t version = read16(stream);
    if (version > VERSION)
    {
        throw std::logic_error("Unsupported NPZ version");
    }

    npy::file_entry entry;
    size_t length = read_shared_header(stream, entry);
    read16(stream); // file comment length
    read16(stream); // disk number start
    read16(stream); // internal file attributes
    read32(stream); // external file attributes
    entry.offset = read32(stream);
    std::vector<char> buffer(length);
    stream.read(buffer.data(), length);
    entry.filename = std::string(buffer.begin(), buffer.end());
    return entry;
}

struct CentralDirectory
{
    std::uint16_t num_entries;
    std::uint32_t size;
    std::uint32_t offset;
};

void write_end_of_central_directory(std::ostream &stream, const CentralDirectory &dir)
{
    stream.write(reinterpret_cast<const char *>(CD_END_SIG.data()), CD_END_SIG.size());
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

CentralDirectory read_end_of_central_directory(std::istream &stream)
{
    assert_sig(stream, CD_END_SIG);

    CentralDirectory result;
    read16(stream); // number of this disk
    read16(stream); // number of the disk with the start of the central directory
    result.num_entries = read16(stream);
    read16(stream); // num_entries_on_disk
    result.size = read32(stream);
    result.offset = read32(stream);
    return result;
}

} // namespace

namespace npy
{
bool file_entry::check(const file_entry &other) const
{
    return !(other.filename != this->filename ||
             other.crc32 != this->crc32 ||
             other.compression_method != this->compression_method ||
             other.compressed_size != this->compressed_size ||
             other.uncompressed_size != this->uncompressed_size);
}

onpzstream::onpzstream(const std::string &path,
                       compression_method method,
                       endian endianness) : m_output(path, std::ios::out | std::ios::binary),
                                            m_compression_method(method),
                                            m_endianness(endianness),
                                            m_closed(false)
{
}

onpzstream::~onpzstream()
{
    if (!this->m_closed)
    {
        this->close();
    }
}

void onpzstream::write_file(const std::string &filename,
                            const std::string &bytes)
{
    std::uint32_t uncompressed_size = static_cast<std::uint32_t>(bytes.length());
    std::uint32_t compressed_size = 0;
    std::string compressed_bytes;
    if (this->m_compression_method == compression_method::STORED)
    {
        compressed_bytes = bytes;
        compressed_size = uncompressed_size;
    }
    else if (this->m_compression_method == compression_method::DEFLATED)
    {
        compressed_bytes = zip::deflate(bytes);
        compressed_size = static_cast<std::uint32_t>(compressed_bytes.length());
    }
    else
    {
        throw std::logic_error("Not implemented");
    }

    file_entry entry = {
        filename,
        zip::crc32(bytes),
        compressed_size,
        uncompressed_size,
        static_cast<std::uint16_t>(this->m_compression_method),
        static_cast<std::uint32_t>(this->m_output.tellp())};
    write_local_header(this->m_output, entry);
    this->m_output.write(compressed_bytes.data(), compressed_size);
    this->m_entries.push_back(std::move(entry));
}

void onpzstream::close()
{
    if (!this->m_closed)
    {
        CentralDirectory dir;
        dir.offset = static_cast<std::uint32_t>(this->m_output.tellp());
        for (auto &header : this->m_entries)
        {
            write_central_directory_header(this->m_output, header);
        }

        dir.size = static_cast<std::uint32_t>(this->m_output.tellp()) - dir.offset;
        dir.num_entries = static_cast<std::uint16_t>(this->m_entries.size());
        write_end_of_central_directory(this->m_output, dir);
        this->m_output.close();
        this->m_closed = true;
    }
}

inpzstream::inpzstream(const std::string &path) : m_input(path, std::ios::out | std::ios::binary)
{
    this->read_entries();
}

void inpzstream::read_entries()
{
    this->m_input.seekg(-CD_END_SIZE, std::ios::end);
    CentralDirectory dir = read_end_of_central_directory(this->m_input);

    this->m_input.seekg(dir.offset, std::ios::beg);

    for (size_t i = 0; i < dir.num_entries; ++i)
    {
        file_entry entry = read_central_directory_header(this->m_input);
        this->m_entries[entry.filename] = entry;
    }
}

std::string inpzstream::read_file(const std::string &filename)
{
    if (this->m_entries.count(filename) == 0)
    {
        throw std::invalid_argument("Key not found");
    }

    const file_entry &entry = this->m_entries[filename];
    this->m_input.seekg(entry.offset, std::ios::beg);

    file_entry local = read_local_header(this->m_input);
    if (!entry.check(local))
    {
        throw std::logic_error("Central directory and local headers disagree");
    }

    std::vector<char> buffer(entry.compressed_size);
    this->m_input.read(buffer.data(), buffer.size());
    compression_method cmethod = static_cast<compression_method>(entry.compression_method);
    std::string bytes = std::string(buffer.begin(), buffer.end());
    std::string uncompressed_bytes = bytes;
    if (cmethod == compression_method::DEFLATED)
    {
        uncompressed_bytes = zip::inflate(bytes);
    }

    std::uint32_t crc32 = zip::crc32(uncompressed_bytes);
    if (crc32 != entry.crc32)
    {
        throw std::logic_error("CRC mismatch");
    }

    return uncompressed_bytes;
}

void inpzstream::close()
{
    this->m_input.close();
}
} // namespace npy