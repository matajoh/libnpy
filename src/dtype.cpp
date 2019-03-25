#include <map>
#include <array>
#include <string>

#include "dtype.h"
#include "util.h"

namespace {
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
        ">f8"
    };

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
        "<f8"
    };

    std::map<std::string, std::pair<npy::data_type, npy::endian>> DTYPE_MAP = {
        {"|u1", {npy::data_type::UINT8, npy::endian::NATIVE}},
        {"|i1", {npy::data_type::INT8, npy::endian::NATIVE}},
        {"<u2", {npy::data_type::UINT16, npy::endian::LITTLE}},
        {">u2", {npy::data_type::UINT16, npy::endian::BIG}},
        {"<i2", {npy::data_type::INT16, npy::endian::LITTLE}},
        {">i2", {npy::data_type::INT16, npy::endian::BIG}},
        {"<u4", {npy::data_type::UINT32, npy::endian::LITTLE}},
        {">u4", {npy::data_type::UINT32, npy::endian::BIG}},
        {"<i4", {npy::data_type::INT32, npy::endian::LITTLE}},
        {">i4", {npy::data_type::INT32, npy::endian::BIG}},
        {"<u8", {npy::data_type::UINT64, npy::endian::LITTLE}},
        {">u8", {npy::data_type::UINT64, npy::endian::BIG}},
        {"<i8", {npy::data_type::INT64, npy::endian::LITTLE}},
        {">i8", {npy::data_type::INT64, npy::endian::BIG}},
        {"<f4", {npy::data_type::FLOAT32, npy::endian::LITTLE}},
        {">f4", {npy::data_type::FLOAT32, npy::endian::BIG}},
        {"<f8", {npy::data_type::FLOAT64, npy::endian::LITTLE}},
        {">f8", {npy::data_type::FLOAT64, npy::endian::BIG}},
    };
}

namespace npy {
    const std::string& to_dtype(data_type dtype, endian endianness) {
        if(endianness == npy::endian::NATIVE){
            endianness = native_endian();
        }

        if(endianness == npy::endian::BIG){
            return BIG_ENDIAN_DTYPES[static_cast<size_t>(dtype)];
        }

        return LITTLE_ENDIAN_DTYPES[static_cast<size_t>(dtype)];
    }

    const std::pair<data_type, endian>& from_dtype(const std::string& dtype) {
        return DTYPE_MAP[dtype];
    }
}