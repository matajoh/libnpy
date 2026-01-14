#include "libnpy_tests.h"
#include <array>
#include <stdexcept>

template <typename T, typename Shape> class custom_tensor {
public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;

  custom_tensor(const Shape shape, bool fortran_order = false)
      : m_shape(shape), m_fortran_order(fortran_order) {
    m_size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      m_size *= shape[i];
    }

    m_data = new T[m_size]{};
  }

  ~custom_tensor() { delete[] m_data; }

  static custom_tensor<T, Shape> load(std::basic_istream<char> &input,
                                      const npy::header_info &info) {
    Shape shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      shape[i] = info.shape[i];
    }

    custom_tensor<T, Shape> result(shape, info.fortran_order);
    npy::read_values(input, result.m_data, result.m_size, info.endianness);
  }

  void save(std::basic_ostream<char> &output, npy::endian_t endianness) const {
    npy::write_values(output, m_data, m_size, endianness);
  }

  T *data() { return m_data; }

  const T *data() const { return m_data; }

  std::size_t size() const { return m_size; }

  size_t shape(size_t i) const { return m_shape[i]; }

  size_t ndim() const { return m_shape.size(); }

  npy::data_type_t dtype() const {
    if constexpr (std::is_same<value_type, float>()) {
      return npy::data_type_t::FLOAT32;
    }

    if constexpr (std::is_same<value_type, double>()) {
      return npy::data_type_t::FLOAT64;
    }

    if constexpr (std::is_same<value_type, int>()) {
      return npy::data_type_t::INT32;
    }

    throw std::invalid_argument("Unsupported matrix type");
  }

  std::string dtype(npy::endian_t endianness) const {
    return npy::to_dtype(dtype(), endianness);
  }

  bool fortran_order() const { return m_fortran_order; }

private:
  T *m_data;
  size_t m_size;
  Shape m_shape;
  bool m_fortran_order;
};

typedef custom_tensor<float, std::array<size_t, 1>> custom1f;
typedef custom_tensor<double, std::array<size_t, 2>> custom2d;
typedef custom_tensor<int, std::array<size_t, 3>> custom3i;

template <typename T> void populate(T &tensor) {
  typename T::pointer ptr = tensor.data();
  for (size_t i = 0; i < tensor.size(); ++i, ++ptr) {
    *ptr = typename T::value_type(i);
  }
}

const std::string TEMP_NPZ = "custom.npz";

int test_custom_tensor() {
  int result = EXIT_SUCCESS;

  custom1f a({3});
  custom2d b{{3, 4}};
  custom3i c({3, 4, 5});

  populate(a);
  populate(b);
  populate(c);

  {
    npy::npzfilewriter npz(TEMP_NPZ);
    npz.write("a", a);
  }

  std::filesystem::remove(TEMP_NPZ);

  return result;
}