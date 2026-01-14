#include "npy/npy.h"
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

/// @brief A custom tensor class which uses Eigen matrices as the backing
/// storage.
/// @details This serves as a more complex example of how to implement a custom
/// tensor type which is compatible with the npy::load and npy::save functions.
template <typename T, int StorageOrder = Eigen::ColMajor> class EigenTensor {
public:
  using MatrixT =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;

  /// @brief Constructor.
  /// @param rows the number of rows
  /// @param columns the number of columns
  EigenTensor(size_t rows, size_t columns) : m_matrix(rows, columns) {}

  /// @brief Constructor.
  /// @param matrix an Eigen matrix to use as the backing storage
  EigenTensor(MatrixT &&matrix) : m_matrix(std::move(matrix)) {}

  /// @brief Constructor.
  /// @param matrix an Eigen matrix to use as the backing storage
  EigenTensor(const MatrixT &matrix) : m_matrix(matrix) {}

  /// @brief Load a tensor from the provided stream.
  /// @details This is one of the methods required by the library to
  /// read NPY files. Note how we handle both row-major and column-major
  /// storage here by using the npy::read_values function appropriately.
  /// @param input the input stream
  /// @param info the header information
  /// @return an instance of the tensor read from the stream
  static EigenTensor<T, StorageOrder> load(std::basic_istream<char> &input,
                                           const npy::header_info &info) {
    if (info.shape.size() > 2) {
      throw std::invalid_argument(
          "Matrices of dimensionality > 2 not supported");
    }

    size_t rows = info.shape[0];
    size_t columns = 1;
    if (info.shape.size() == 2) {
      columns = info.shape[1];
    }

    MatrixT result(rows, columns);

    if (info.fortran_order) {
      if (StorageOrder == Eigen::ColMajor || columns == 1) {
        npy::read_values(input, result.data(), rows * columns, info);
      } else {
        /// Copy into a temporary column-major matrix and then assign to the
        /// row-major result.
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> cm(
            rows, columns);
        npy::read_values(input, cm.data(), rows * columns, info);
        result = cm;
      }
    } else {
      if (StorageOrder == Eigen::RowMajor || columns == 1) {
        npy::read_values(input, result.data(), rows * columns, info);
      } else {
        /// Copy into a temporary row-major matrix and then assign to the
        /// column-major result.
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rm(
            rows, columns);
        npy::read_values(input, rm.data(), rows * columns, info);
        result = rm;
      }
    }

    return result;
  }

  /// @brief Save the tensor to the provided stream.
  /// @details This is one of the methods required by the library to
  /// write NPY files.
  void save(std::basic_ostream<char> &output, npy::endian_t endianness) const {
    npy::write_values(output, data(), size(), endianness);
  }

  /// @brief Return the number of dimensions of the tensor.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @return the number of dimensions (2)
  size_t ndim() const { return 2; }

  /// @brief Returns the dimensionality of the tensor at the specified index.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @param index index into the shape
  /// @return the dimensionality at the index
  size_t shape(int index) const {
    switch (index) {
    case 0:
      return m_matrix.rows();

    case 1:
      return m_matrix.cols();

    default:
      throw std::invalid_argument(
          "Matrix only has two dimensions (rows, columns)");
    }
  }

  /// @brief The number of elements in the tensor.
  /// @return the number of elements
  size_t size() const { return m_matrix.rows() * m_matrix.cols(); }

  /// @brief Whether the tensor data is stored in FORTRAN, or column-major,
  /// order.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @return whether the tensor is stored in FORTRAN order
  bool fortran_order() const { return StorageOrder == Eigen::ColMajor; }

  /// @brief A pointer to the start of the underlying values buffer.
  /// @return a pointer to the data
  T *data() { return m_matrix.data(); }

  /// @brief A pointer to the start of the underlying values buffer.
  /// @return a pointer to the data
  const T *data() const { return m_matrix.data(); }

  /// @brief The data type of the tensor.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files.
  /// @return the data type
  npy::data_type_t dtype() const {
    if constexpr (std::is_same<T, float>()) {
      return npy::data_type_t::FLOAT32;
    }

    if constexpr (std::is_same<T, double>()) {
      return npy::data_type_t::FLOAT64;
    }

    throw std::invalid_argument("Unsupported matrix type");
  }

  /// @brief The data type of the tensor as a dtype string.
  /// @details This is one of the methods required by the library to
  /// read and write NPY files. Note that you can use the npy::to_dtype
  /// function to convert a data_type_t into a dtype string with the desired
  /// endianness.
  /// @param endianness the endianness to use in the dtype string
  /// @return the data type string
  std::string dtype(npy::endian_t endianness) const {
    return npy::to_dtype(dtype(), endianness);
  }

  /// @brief Access the underlying Eigen matrix.
  /// @return a reference to the Eigen matrix
  MatrixT &matrix() { return m_matrix; }

  /// @brief Access the underlying Eigen matrix.
  /// @return a const reference to the Eigen matrix
  const MatrixT &matrix() const { return m_matrix; }

private:
  MatrixT m_matrix;
};

int main() {
  EigenTensor<float> a(Eigen::MatrixXf::Random(256, 256));
  EigenTensor<float> b(Eigen::MatrixXf::Random(256, 512));
  auto ab = a.matrix() * b.matrix();
  // Create a double-precision version of the result in row-major order
  EigenTensor<double, Eigen::RowMajor> c(ab.cast<double>());

  {
    npy::npzfilewriter npz("custom.npz");
    npz.write("a", a);
    npz.write("b", b);
    npz.write("c", c);
  }

  npy::npzfilereader npz("custom.npz");
  EigenTensor<float> a_out = npz.read<EigenTensor<float>>("a");
  EigenTensor<float> b_out = npz.read<EigenTensor<float>>("b");
  // read the result back out, but in column-major order this time
  EigenTensor<double> c_out = npz.read<EigenTensor<double>>("c");

  if (a.matrix() != a_out.matrix()) {
    std::cout << "a and a_out differ!" << std::endl;
    return 1;
  }

  if (b.matrix() != b_out.matrix()) {
    std::cout << "b and b_out differ!" << std::endl;
    return 1;
  }

  if (c.matrix() != c_out.matrix()) {
    std::cout << "c and c_out differ!" << std::endl;
    return 1;
  }

  return 0;
}