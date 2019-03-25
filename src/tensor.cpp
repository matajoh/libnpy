#include "tensor.h"

namespace npy {
    template<>
    data_type tensor<std::int8_t>::get_dtype(){
        return data_type::INT8;
    };

    template<>
    data_type tensor<std::uint8_t>::get_dtype(){
        return data_type::UINT8;
    };

    template<>
    data_type tensor<std::int16_t>::get_dtype(){
        return data_type::INT16;
    };

    template<>
    data_type tensor<std::uint16_t>::get_dtype(){
        return data_type::UINT16;
    };

    template<>
    data_type tensor<std::int32_t>::get_dtype(){
        return data_type::INT32;
    };

    template<>
    data_type tensor<std::uint32_t>::get_dtype(){
        return data_type::UINT32;
    };

    template<>
    data_type tensor<std::int64_t>::get_dtype(){
        return data_type::INT64;
    };

    template<>
    data_type tensor<std::uint64_t>::get_dtype(){
        return data_type::UINT64;
    };

    template<>
    data_type tensor<float>::get_dtype(){
        return data_type::FLOAT32;
    };

    template<>
    data_type tensor<double>::get_dtype(){
        return data_type::FLOAT64;
    };
}