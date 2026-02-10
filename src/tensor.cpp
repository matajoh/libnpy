#include "npy/npy.h"
#include <complex>

namespace npy {
template <> data_type_t tensor<std::int8_t>::get_dtype() {
  return data_type_t::INT8;
};

template <> data_type_t tensor<std::uint8_t>::get_dtype() {
  return data_type_t::UINT8;
};

template <> data_type_t tensor<std::int16_t>::get_dtype() {
  return data_type_t::INT16;
};

template <> data_type_t tensor<std::uint16_t>::get_dtype() {
  return data_type_t::UINT16;
};

template <> data_type_t tensor<std::int32_t>::get_dtype() {
  return data_type_t::INT32;
};

template <> data_type_t tensor<std::uint32_t>::get_dtype() {
  return data_type_t::UINT32;
};

template <> data_type_t tensor<std::int64_t>::get_dtype() {
  return data_type_t::INT64;
};

template <> data_type_t tensor<std::uint64_t>::get_dtype() {
  return data_type_t::UINT64;
};

template <> data_type_t tensor<float>::get_dtype() {
  return data_type_t::FLOAT32;
};

template <> data_type_t tensor<double>::get_dtype() {
  return data_type_t::FLOAT64;
};

template <> data_type_t tensor<std::complex<float>>::get_dtype() {
  return data_type_t::COMPLEX64;
}

template <> data_type_t tensor<std::complex<double>>::get_dtype() {
  return data_type_t::COMPLEX128;
}

template <> data_type_t tensor<std::wstring>::get_dtype() {
  return data_type_t::UNICODE_STRING;
}

template <> data_type_t tensor<boolean>::get_dtype() {
  return data_type_t::BOOL;
}

} // namespace npy