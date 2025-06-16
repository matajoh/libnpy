#include "npy/npz.h"

void write_header_info(const npy::header_info &header,
                       const std::string &tab = "") {
  std::cout << tab
            << "Data Type: " << npy::to_dtype(header.dtype, header.endianness)
            << std::endl;
  std::cout << tab << "Fortran Order: " << (header.fortran_order ? "Yes" : "No")
            << std::endl;
  std::cout << tab << "Shape: (";
  if (header.shape.empty()) {
    std::cout << ")";
  } else if (header.shape.size() == 1) {
    std::cout << header.shape[0] << ",)";
  } else {
    std::cout << header.shape[0];
    if (header.shape.size() > 1) {
      for (size_t i = 1; i < header.shape.size(); ++i) {
        std::cout << ", " << header.shape[i];
      }
    }
    std::cout << ")";
  }
  std::cout << std::endl;
  if (header.dtype == npy::data_type_t::UNICODE_STRING) {
    std::cout << tab << "Max Element Length: " << header.max_element_length
              << std::endl;
  }
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename.npy>" << std::endl;
    return 1;
  }

  const std::string filename(argv[1]);
  const std::string ext = filename.substr(filename.size() - 4);
  if (ext == ".npy") {
    // Peek at the NPY file header
    write_header_info(npy::peek(filename));
  } else if (ext == ".npz") {
    // Peek at the NPZ file
    npy::inpzstream input(filename);
    std::cout << "NPZ File Contents:" << std::endl;
    for (const auto &key : input.keys()) {
      npy::header_info header = input.peek(key);
      std::cout << "Key: " << key << std::endl;
      write_header_info(header, "  ");
    }
  } else {
    std::cerr << "Unsupported file format: " << ext << std::endl;
    return 1;
  }
}