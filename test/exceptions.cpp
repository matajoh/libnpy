#include "libnpy_tests.h"
#include "npz.h"

namespace
{
void save_invalid_path(){
    npy::tensor<std::uint8_t> tensor = test::test_tensor<std::uint8_t>({5, 2, 5});
    npy::save(test::path_join({"does_not_exist", "bad.npy"}), tensor);
}

void load_invalid_path(){
    npy::load<std::uint8_t, npy::tensor>(test::path_join({"does_not_exist", "bad.npy"}));
}

void onpzstream_invalid_path(){
    npy::onpzstream(test::path_join({"does_not_exist", "bad.npz"}));
}

void inpzstream_invalid_path(){
    npy::inpzstream(test::path_join({"does_not_exist", "bad.npz"}));
}

void inpzstream_compression(){
}
} // namespace

int test_exceptions()
{
    int result = EXIT_SUCCESS;

    test::assert_throws<std::invalid_argument>(save_invalid_path, result, "save_invalid_path");    
    test::assert_throws<std::invalid_argument>(load_invalid_path, result, "load_invalid_path");
    test::assert_throws<std::invalid_argument>(onpzstream_invalid_path, result, "onpzstream_invalid_path");


    test::assert_throws<std::logic_error>(inpzstream_invalid_path, result, "inpzstream_invalid_path");

    return result;
}