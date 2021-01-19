#include <map>
#include <array>
#include <string>

#include "npy/core.h"

namespace
{
std::array<std::string, 11> BIG_ENDIAN_DTYPES = {
    "|i1",
    "|u1",
    ">i2",
    ">u2",
    ">i4",
    ">u4",
    ">i8",
    ">u8",
    ">f4",
    ">f8"};

std::array<std::string, 11> LITTLE_ENDIAN_DTYPES = {
    "|i1",
    "|u1",
    "<i2",
    "<u2",
    "<i4",
    "<u4",
    "<i8",
    "<u8",
    "<f4",
    "<f8"};

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
};
} // namespace

namespace npy
{
const std::string &to_dtype(data_type_t dtype, endian_t endianness)
{
    if (endianness == npy::endian_t::NATIVE)
    {
        endianness = native_endian();
    }

    if (endianness == npy::endian_t::BIG)
    {
        return BIG_ENDIAN_DTYPES[static_cast<size_t>(dtype)];
    }

    return LITTLE_ENDIAN_DTYPES[static_cast<size_t>(dtype)];
}

const std::pair<data_type_t, endian_t> &from_dtype(const std::string &dtype)
{
    return DTYPE_MAP[dtype];
}

std::ostream &operator<<(std::ostream &os, const data_type_t &value)
{
    os << static_cast<int>(value);
    return os;
}

std::ostream &operator<<(std::ostream &os, const endian_t &value)
{
    os << static_cast<int>(value);
    return os;
}

} // namespace npy