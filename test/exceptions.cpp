#include "libnpy_tests.h"
#include "npy.h"
#include "npz.h"
#include "tensor.h"

namespace
{
npy::tensor<std::uint8_t> TENSOR(std::vector<size_t>({ 5, 2, 5 }));

void save_invalid_path()
{
    npy::save(test::path_join({"does_not_exist", "bad.npy"}), TENSOR);
}

void load_invalid_path()
{
    npy::load<std::uint8_t, npy::tensor>(test::path_join({"does_not_exist", "bad.npy"}));
}

void peek_invalid_path()
{
    npy::peek(test::path_join({"does_not_exist", "bad.npy"}));
}

void inpzstream_invalid_path()
{
    npy::inpzstream(test::path_join({"does_not_exist", "bad.npz"}));
}

void inpzstream_read_invalid_filename()
{
    npy::inpzstream stream(test::path_join({"assets", "test", "test.npz"}));
    npy::tensor<std::uint8_t> tensor = stream.read<std::uint8_t>("not_there.npy");
}

void inpzstream_peek_invalid_filename()
{
    npy::inpzstream stream(test::path_join({"assets", "test", "test.npz"}));
    npy::header_info header = stream.peek("not_there.npy");
}

void onpzstream_compression()
{
	npy::compression_method_t compression_method = static_cast<npy::compression_method_t>(99);
	npy::onpzstream stream("test.npz", compression_method);
	stream.write("test.npy", TENSOR);
}

void tensor_copy_from_0()
{
    std::vector<std::uint8_t> buffer;
	TENSOR.copy_from(buffer.data(), buffer.size());
}

void tensor_copy_from_1()
{
    std::vector<std::uint8_t> buffer;
	TENSOR.copy_from(buffer);
}

void tensor_move_from()
{
    std::vector<std::uint8_t> buffer;
	TENSOR.copy_from(std::move(buffer));
}

void tensor_index_size()
{
    std::uint8_t value = TENSOR({0, 0});
}

void tensor_index_range()
{
    std::uint8_t value = TENSOR({2, 3, 3});
}

void load_wrong_dtype()
{
    npy::tensor<float> tensor = npy::load<float, npy::tensor>(test::path_join({"assets", "test", "uint8.npy"}));
}

void onpzstream_closed()
{
    npy::onpzstream stream("test.npz");
    stream.close();
    stream.write("error.npy", TENSOR);
}

void inpzstream_invalid_file()
{
    npy::inpzstream stream(test::path_join({"assets", "test", "uint8.npy"}));
}

} // namespace

int test_exceptions()
{
    int result = EXIT_SUCCESS;

    test::assert_throws<std::invalid_argument>(peek_invalid_path, result, "peek_invalid_path");
    test::assert_throws<std::invalid_argument>(save_invalid_path, result, "save_invalid_path");
    test::assert_throws<std::invalid_argument>(load_invalid_path, result, "load_invalid_path");
    test::assert_throws<std::invalid_argument>(inpzstream_invalid_path, result, "inpzstream_invalid_path");
    test::assert_throws<std::invalid_argument>(inpzstream_read_invalid_filename, result, "inpzstream_read_invalid_filename");
    test::assert_throws<std::invalid_argument>(inpzstream_peek_invalid_filename, result, "inpzstream_peek_invalid_filename");
	test::assert_throws<std::invalid_argument>(onpzstream_compression, result, "onpzstream_compression");
	test::assert_throws<std::invalid_argument>(tensor_copy_from_0, result, "tensor_copy_from_0");
    test::assert_throws<std::invalid_argument>(tensor_copy_from_1, result, "tensor_copy_from_1");
    test::assert_throws<std::invalid_argument>(tensor_move_from, result, "tensor_move_from");
    test::assert_throws<std::invalid_argument>(tensor_index_size, result, "tensor_index");

    test::assert_throws<std::out_of_range>(tensor_index_range, result, "tensor_index_range");

    test::assert_throws<std::logic_error>(load_wrong_dtype, result, "load_wrong_dtype");
    test::assert_throws<std::logic_error>(onpzstream_closed, result, "onpzstream_closed");
    test::assert_throws<std::logic_error>(inpzstream_invalid_file, result, "inpzstream_invalid_file");

    std::remove("test.npz");

    return result;
}