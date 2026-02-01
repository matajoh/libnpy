#include <array>
#include <complex>
#include <map>
#include <stdexcept>
#include <string>

#include "npy/npy.h"

#define GETC(x) static_cast<char>((x).get())

namespace {
std::array<std::string, 13> BIG_ENDIAN_DTYPES = {"|i1", "|u1", ">i2", ">u2",
                                                 ">i4", ">u4", ">i8", ">u8",
                                                 ">f4", ">f8", ">c8", ">c16",
                                                 "|b1"};

std::array<std::string, 13> LITTLE_ENDIAN_DTYPES = {
    "|i1", "|u1", "<i2", "<u2", "<i4", "<u4",
    "<i8", "<u8", "<f4", "<f8", "<c8", "<c16",
    "|b1"};

std::map<std::string, std::pair<npy::data_type_t, npy::endian_t>> DTYPE_MAP = {
    {"|u1", {npy::data_type_t::UINT8, npy::endian_t::NATIVE}},
    {"|i1", {npy::data_type_t::INT8, npy::endian_t::NATIVE}},
    {"<u2", {npy::data_type_t::UINT16, npy::endian_t::LITTLE}},
    {">u2", {npy::data_type_t::UINT16, npy::endian_t::BIG}},
    {"<i2", {npy::data_type_t::INT16, npy::endian_t::LITTLE}},
    {">i2", {npy::data_type_t::INT16, npy::endian_t::BIG}},
    {"<u4", {npy::data_type_t::UINT32, npy::endian_t::LITTLE}},
    {">u4", {npy::data_type_t::UINT32, npy::endian_t::BIG}},
    {"<i4", {npy::data_type_t::INT32, npy::endian_t::LITTLE}},
    {">i4", {npy::data_type_t::INT32, npy::endian_t::BIG}},
    {"<u8", {npy::data_type_t::UINT64, npy::endian_t::LITTLE}},
    {">u8", {npy::data_type_t::UINT64, npy::endian_t::BIG}},
    {"<i8", {npy::data_type_t::INT64, npy::endian_t::LITTLE}},
    {">i8", {npy::data_type_t::INT64, npy::endian_t::BIG}},
    {"<f4", {npy::data_type_t::FLOAT32, npy::endian_t::LITTLE}},
    {">f4", {npy::data_type_t::FLOAT32, npy::endian_t::BIG}},
    {"<f8", {npy::data_type_t::FLOAT64, npy::endian_t::LITTLE}},
    {">f8", {npy::data_type_t::FLOAT64, npy::endian_t::BIG}},
    {"<c8", {npy::data_type_t::COMPLEX64, npy::endian_t::LITTLE}},
    {">c8", {npy::data_type_t::COMPLEX64, npy::endian_t::BIG}},
    {"<c16", {npy::data_type_t::COMPLEX128, npy::endian_t::LITTLE}},
    {">c16", {npy::data_type_t::COMPLEX128, npy::endian_t::BIG}},
    {"|b1", {npy::data_type_t::BOOL, npy::endian_t::NATIVE}}};
} // namespace

namespace npy {
const std::string &to_dtype(data_type_t dtype, endian_t endianness) {
  if (dtype == data_type_t::UNICODE_STRING) {
    throw std::invalid_argument("U dtype must be computed dynamically");
  }

  if (endianness == npy::endian_t::NATIVE) {
    endianness = native_endian();
  }

  if (endianness == npy::endian_t::BIG) {
    return BIG_ENDIAN_DTYPES[static_cast<size_t>(dtype)];
  }

  return LITTLE_ENDIAN_DTYPES[static_cast<size_t>(dtype)];
}

const std::pair<data_type_t, endian_t> &from_dtype(const std::string &dtype) {
  return DTYPE_MAP[dtype];
}

std::ostream &operator<<(std::ostream &os, const data_type_t &value) {
  os << static_cast<int>(value);
  return os;
}

std::ostream &operator<<(std::ostream &os, const endian_t &value) {
  os << static_cast<int>(value);
  return os;
}

template <>
void write_values<>(std::basic_ostream<char> &output, const uint8_t *data_ptr,
                    size_t num_elements, endian_t) {
  output.write(reinterpret_cast<const char *>(data_ptr), num_elements);
}

template <>
void read_values<>(std::basic_istream<char> &input, uint8_t *data_ptr,
                   size_t num_elements, const header_info &) {
  char *start = reinterpret_cast<char *>(data_ptr);
  input.read(start, num_elements);
}

template <>
void write_values<>(std::basic_ostream<char> &output, const int8_t *data_ptr,
                    size_t num_elements, endian_t) {
  output.write(reinterpret_cast<const char *>(data_ptr), num_elements);
}

template <>
void read_values<>(std::basic_istream<char> &input, int8_t *data_ptr,
                   size_t num_elements, const header_info &) {
  char *start = reinterpret_cast<char *>(data_ptr);
  input.read(start, num_elements);
}

template <>
void write_values<>(std::basic_ostream<char> &output, const bool *data_ptr,
                    size_t num_elements, endian_t) {
  for (size_t i = 0; i < num_elements; ++i) {
    char byte = data_ptr[i] ? 1 : 0;
    output.write(&byte, 1);
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, bool *data_ptr,
                   size_t num_elements, const header_info &) {
  for (size_t i = 0; i < num_elements; ++i) {
    char byte;
    input.read(&byte, 1);
    data_ptr[i] = (byte != 0);
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const uint_least16_t *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 2);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, uint_least16_t *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 2);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 2) {
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const int_least16_t *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 2);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, int_least16_t *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 2);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 2) {
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const uint_least32_t *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 4);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, uint_least32_t *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 4);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 4) {
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const int_least32_t *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 4);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, int_least32_t *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 4);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 4) {
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output, const float *data_ptr,
                    size_t num_elements, endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 4);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, float *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 4);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 4) {
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const uint_least64_t *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 8);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[7]);
      output.put(start[6]);
      output.put(start[5]);
      output.put(start[4]);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, uint_least64_t *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 8);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 8) {
      ptr[7] = GETC(input);
      ptr[6] = GETC(input);
      ptr[5] = GETC(input);
      ptr[4] = GETC(input);
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const int_least64_t *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 8);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[7]);
      output.put(start[6]);
      output.put(start[5]);
      output.put(start[4]);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, int_least64_t *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 8);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 8) {
      ptr[7] = GETC(input);
      ptr[6] = GETC(input);
      ptr[5] = GETC(input);
      ptr[4] = GETC(input);
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output, const double *data_ptr,
                    size_t num_elements, endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 8);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[7]);
      output.put(start[6]);
      output.put(start[5]);
      output.put(start[4]);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, double *data_ptr,
                   size_t num_elements, const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 8);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 8) {
      ptr[7] = GETC(input);
      ptr[6] = GETC(input);
      ptr[5] = GETC(input);
      ptr[4] = GETC(input);
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const std::complex<float> *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 8);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[7]);
      output.put(start[6]);
      output.put(start[5]);
      output.put(start[4]);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input,
                   std::complex<float> *data_ptr, size_t num_elements,
                   const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 8);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 8) {
      ptr[7] = GETC(input);
      ptr[6] = GETC(input);
      ptr[5] = GETC(input);
      ptr[4] = GETC(input);
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const std::complex<double> *data_ptr, size_t num_elements,
                    endian_t endianness) {
  if (endianness == npy::endian_t::NATIVE || endianness == native_endian()) {
    output.write(reinterpret_cast<const char *>(data_ptr), num_elements * 16);
  } else {
    for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
      const char *start = reinterpret_cast<const char *>(curr);
      output.put(start[15]);
      output.put(start[14]);
      output.put(start[13]);
      output.put(start[12]);
      output.put(start[11]);
      output.put(start[10]);
      output.put(start[9]);
      output.put(start[8]);
      output.put(start[7]);
      output.put(start[6]);
      output.put(start[5]);
      output.put(start[4]);
      output.put(start[3]);
      output.put(start[2]);
      output.put(start[1]);
      output.put(start[0]);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input,
                   std::complex<double> *data_ptr, size_t num_elements,
                   const header_info &info) {
  char *start = reinterpret_cast<char *>(data_ptr);
  if (info.endianness == npy::endian_t::NATIVE ||
      info.endianness == native_endian()) {
    input.read(start, num_elements * 16);
  } else {
    char *ptr = start;
    for (size_t i = 0; i < num_elements; ++i, ptr += 16) {
      ptr[15] = GETC(input);
      ptr[14] = GETC(input);
      ptr[13] = GETC(input);
      ptr[12] = GETC(input);
      ptr[11] = GETC(input);
      ptr[10] = GETC(input);
      ptr[9] = GETC(input);
      ptr[8] = GETC(input);
      ptr[7] = GETC(input);
      ptr[6] = GETC(input);
      ptr[5] = GETC(input);
      ptr[4] = GETC(input);
      ptr[3] = GETC(input);
      ptr[2] = GETC(input);
      ptr[1] = GETC(input);
      ptr[0] = GETC(input);
    }
  }
}

template <>
void write_values<>(std::basic_ostream<char> &output,
                    const std::wstring *data_ptr, size_t num_elements,
                    endian_t endianness) {
  size_t max_element_length = 0;
  for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
    if (curr->size() > max_element_length) {
      max_element_length = curr->size();
    }
  }

  for (auto curr = data_ptr; curr < data_ptr + num_elements; ++curr) {
    for (size_t i = 0; i < curr->size(); ++i) {
      std::int_least32_t value = static_cast<std::int_least32_t>(curr->at(i));

      const char *start = reinterpret_cast<const char *>(&value);
      if (endianness == npy::endian_t::NATIVE ||
          endianness == native_endian()) {
        output.put(start[0]);
        output.put(start[1]);
        output.put(start[2]);
        output.put(start[3]);
      } else {
        output.put(start[3]);
        output.put(start[2]);
        output.put(start[1]);
        output.put(start[0]);
      }
    }

    for (size_t i = curr->size(); i < max_element_length; ++i) {
      output.put(0);
      output.put(0);
      output.put(0);
      output.put(0);
    }
  }
}

template <>
void read_values<>(std::basic_istream<char> &input, std::wstring *data_ptr,
                   size_t num_elements, const header_info &info) {
  std::wstring *ptr = data_ptr;
  for (size_t i = 0; i < num_elements; ++i, ++ptr) {
    std::int_least32_t value = 0;
    char *bytes = reinterpret_cast<char *>(&value);
    for (size_t j = 0; j < info.max_element_length; ++j) {
      if (info.endianness == npy::endian_t::NATIVE ||
          info.endianness == native_endian()) {
        bytes[0] = GETC(input);
        bytes[1] = GETC(input);
        bytes[2] = GETC(input);
        bytes[3] = GETC(input);
      } else {
        bytes[3] = GETC(input);
        bytes[2] = GETC(input);
        bytes[1] = GETC(input);
        bytes[0] = GETC(input);
      }
      if (value == 0) {
        continue;
      }

      ptr->push_back(static_cast<wchar_t>(value));
    }
  }
}

} // namespace npy