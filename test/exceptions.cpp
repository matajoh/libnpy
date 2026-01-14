#include "libnpy_tests.h"
#include <stdexcept>

namespace {

void save_invalid_path(npy::tensor<std::uint8_t> &tensor) {
  npy::save(test::path_join({"does_not_exist", "bad.npy"}), tensor);
}

void load_invalid_path() {
  npy::load<npy::tensor<std::uint8_t>>(
      test::path_join({"does_not_exist", "bad.npy"}));
}

void peek_invalid_path() {
  npy::peek(test::path_join({"does_not_exist", "bad.npy"}));
}

void npzfilereader_invalid_path() {
  npy::npzfilereader(test::path_join({"does_not_exist", "bad.npz"}));
}

void npzfilereader_read_invalid_filename() {
  npy::npzfilereader stream(test::path_join({"assets", "test", "test.npz"}));
  npy::tensor<std::uint8_t> tensor =
      stream.read<npy::tensor<std::uint8_t>>("not_there.npy");
}

void npzfilereader_peek_invalid_filename() {
  npy::npzfilereader stream(test::path_join({"assets", "test", "test.npz"}));
  npy::header_info header = stream.peek("not_there.npy");
}

void npzfilewriter_compression(npy::tensor<std::uint8_t> &tensor) {
  npy::compression_method_t compression_method =
      static_cast<npy::compression_method_t>(99);
  npy::npzfilewriter stream("test.npz", compression_method);
  stream.write("test.npy", tensor);
}

void tensor_copy_from_0(npy::tensor<std::uint8_t> &tensor) {
  std::vector<std::uint8_t> buffer;
  tensor.copy_from(buffer.data(), buffer.size());
}

void tensor_copy_from_1(npy::tensor<std::uint8_t> &tensor) {
  std::vector<std::uint8_t> buffer;
  tensor.copy_from(buffer);
}

void tensor_move_from(npy::tensor<std::uint8_t> &tensor) {
  std::vector<std::uint8_t> buffer;
  tensor.move_from(std::move(buffer));
}

void tensor_index_size(npy::tensor<std::uint8_t> &tensor) {
  std::uint8_t value = tensor(0, 0);
}

void tensor_index_range(npy::tensor<std::uint8_t> &tensor) {
  std::uint8_t value = tensor(2, 3, 3);
}

void load_wrong_dtype() {
  npy::tensor<float> tensor = npy::load<npy::tensor<float>>(
      test::path_join({"assets", "test", "uint8.npy"}));
}

void npzfilewriter_closed(npy::tensor<std::uint8_t> &tensor) {
  npy::npzfilewriter stream("test.npz");
  stream.close();
  stream.write("error.npy", tensor);
}

void npzfilereader_invalid_file() {
  npy::npzfilereader stream(test::path_join({"assets", "test", "uint8.npy"}));
}

typedef npy::tensor<std::uint8_t> tensor_t;

} // namespace

int test_exceptions() {
  int result = EXIT_SUCCESS;

  tensor_t tensor({5, 2, 5});

  test::assert_throws<std::invalid_argument>(peek_invalid_path, result,
                                             "peek_invalid_path");
  test::assert_throws<std::invalid_argument, tensor_t &>(
      save_invalid_path, tensor, result, "save_invalid_path");
  test::assert_throws<std::invalid_argument>(load_invalid_path, result,
                                             "load_invalid_path");
  test::assert_throws<std::invalid_argument>(npzfilereader_invalid_path, result,
                                             "npzfilereader_invalid_path");
  test::assert_throws<std::invalid_argument>(
      npzfilereader_read_invalid_filename, result,
      "npzfilereader_read_invalid_filename");
  test::assert_throws<std::invalid_argument>(
      npzfilereader_peek_invalid_filename, result,
      "npzfilereader_peek_invalid_filename");
  test::assert_throws<std::invalid_argument, tensor_t &>(
      npzfilewriter_compression, tensor, result, "npzfilewriter_compression");
  test::assert_throws<std::invalid_argument, tensor_t &>(
      tensor_copy_from_0, tensor, result, "tensor_copy_from_0");
  test::assert_throws<std::invalid_argument, tensor_t &>(
      tensor_copy_from_1, tensor, result, "tensor_copy_from_1");
  test::assert_throws<std::invalid_argument, tensor_t &>(
      tensor_move_from, tensor, result, "tensor_move_from");
  test::assert_throws<std::invalid_argument, tensor_t &>(
      tensor_index_size, tensor, result, "tensor_index");

  test::assert_throws<std::invalid_argument, tensor_t &>(
      tensor_index_range, tensor, result, "tensor_index_range");

  test::assert_throws<std::runtime_error>(load_wrong_dtype, result,
                                          "load_wrong_dtype");
  test::assert_throws<std::runtime_error, tensor_t &>(
      npzfilewriter_closed, tensor, result, "npzfilewriter_closed");
  test::assert_throws<std::runtime_error>(npzfilereader_invalid_file, result,
                                          "npzfilereader_invalid_file");

  std::remove("test.npz");

  return result;
}