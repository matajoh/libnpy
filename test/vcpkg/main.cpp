#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <npy/npy.h>

int main() {
  const std::string path = "vcpkg_test.npy";

  // Create a small 2x3 tensor of float32
  npy::tensor<float> original({2, 3});
  for (size_t i = 0; i < original.size(); ++i) {
    original.data()[i] = static_cast<float>(i) * 1.5f;
  }

  // Save to disk
  npy::save(path, original);

  // Load back
  auto loaded = npy::load<float, npy::tensor>(path);

  // Verify shape
  if (loaded.shape() != original.shape()) {
    std::cerr << "Shape mismatch" << std::endl;
    std::filesystem::remove(path);
    return EXIT_FAILURE;
  }

  // Verify data
  for (size_t i = 0; i < original.size(); ++i) {
    if (loaded.data()[i] != original.data()[i]) {
      std::cerr << "Data mismatch at index " << i << std::endl;
      std::filesystem::remove(path);
      return EXIT_FAILURE;
    }
  }

  std::filesystem::remove(path);
  std::cout << "vcpkg integration test passed" << std::endl;
  return EXIT_SUCCESS;
}
