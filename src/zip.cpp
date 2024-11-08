#include "miniz/miniz.h"
#include <cassert>
#include <stdexcept>

#include "npy/core.h"
#include "zip.h"

namespace {
const size_t CHUNK = 1024 * 1024;
const int WINDOW_BITS = -15;
const int MEM_LEVEL = 8;
} // namespace

namespace npy {

std::uint32_t npy_crc32(const std::string &bytes) {
  uLong crc = ::crc32(0L, Z_NULL, 0);
  const Bytef *buf = reinterpret_cast<const Bytef *>(bytes.data());
  uInt len = static_cast<uInt>(bytes.size());
  return ::crc32(crc, buf, len);
}

std::string npy_deflate(std::string &&bytes) {
  int ret, flush;
  unsigned have;
  z_stream strm;
  std::vector<unsigned char> in(CHUNK);
  std::vector<unsigned char> out(CHUNK);

  /* allocate deflate state */
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;
  ret = deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, WINDOW_BITS,
                     MEM_LEVEL, Z_DEFAULT_STRATEGY);
  if (ret != Z_OK) {
    throw std::logic_error("Unable to initialize deflate algorithm");
  }

  std::string str(bytes.begin(), bytes.end());
  imemstream input(std::move(str));
  omemstream output;

  /* compress until end of file */
  do {
    input.read(reinterpret_cast<char *>(in.data()), CHUNK);
    strm.avail_in = static_cast<uInt>(input.gcount());
    if (input.eof()) {
      flush = Z_FINISH;
    } else {
      if (input.fail() || input.bad()) {
        (void)deflateEnd(&strm);
        throw std::logic_error("Error reading from input stream");
      }

      flush = Z_NO_FLUSH;
    }

    strm.next_in = in.data();

    /* run deflate() on input until output buffer not full, finish
       compression if all of source has been read in */
    do {
      strm.avail_out = CHUNK;
      strm.next_out = out.data();
      ret = deflate(&strm, flush);
      assert(ret != Z_STREAM_ERROR); /* state not clobbered */
      have = CHUNK - strm.avail_out;
      output.write(reinterpret_cast<char *>(out.data()), have);
      if (output.fail() || output.bad()) {
        (void)deflateEnd(&strm);
        throw std::logic_error("Error writing to output stream");
      }
    } while (strm.avail_out == 0);
    assert(strm.avail_in == 0); /* all input will be used */
                                /* done when last data in file processed */
  } while (flush != Z_FINISH);
  assert(ret == Z_STREAM_END); /* stream will be complete */

  /* clean up and return */
  (void)deflateEnd(&strm);

  return output.str();
}

std::string npy_inflate(std::string &&bytes) {
  int ret;
  unsigned have;
  z_stream strm;
  std::vector<unsigned char> in(CHUNK);
  std::vector<unsigned char> out(CHUNK);
  /* allocate inflate state */
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;
  strm.avail_in = 0;
  strm.next_in = Z_NULL;
  ret = inflateInit2(&strm, WINDOW_BITS);
  if (ret != Z_OK) {
    throw std::logic_error("Unable to initialize inflate algorithm");
  }

  std::string str(bytes.begin(), bytes.end());
  imemstream input(std::move(str));
  omemstream output;

  /* decompress until deflate stream ends or end of file */
  do {
    input.read(reinterpret_cast<char *>(in.data()), CHUNK);
    strm.avail_in = static_cast<uInt>(input.gcount());
    if ((input.fail() && !input.eof()) || input.bad()) {
      (void)inflateEnd(&strm);
      throw std::logic_error("Error reading from input stream");
    }

    if (strm.avail_in == 0) {
      break;
    }

    strm.next_in = in.data();

    /* run inflate() on input until output buffer not full */
    do {
      strm.avail_out = CHUNK;
      strm.next_out = out.data();
      ret = inflate(&strm, Z_NO_FLUSH);
      assert(ret != Z_STREAM_ERROR); /* state not clobbered */
      switch (ret) {
      case Z_NEED_DICT:
        ret = Z_DATA_ERROR; /* and fall through */
      case Z_DATA_ERROR:
      case Z_MEM_ERROR:
        (void)inflateEnd(&strm);
        throw std::logic_error("Error inflating stream");
      }

      have = CHUNK - strm.avail_out;
      output.write(reinterpret_cast<char *>(out.data()), have);
      if (output.fail() || output.bad()) {
        (void)inflateEnd(&strm);
        throw std::logic_error("Error writing to output stream");
      }
    } while (strm.avail_out == 0);

    /* done when inflate() says it's done */
  } while (ret != Z_STREAM_END);

  /* clean up and return */
  (void)inflateEnd(&strm);

  if (ret == Z_STREAM_END) {
    return output.str();
  }

  throw std::logic_error("Error inflating stream");
}

} // namespace npy