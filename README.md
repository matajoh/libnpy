# libnpy

`libnpy` is a multi-platform C++ library for reading and writing NPY and
NPZ files, with an additional .NET interface. It was built with the 
intention of making it easier for multi-language projects to use NPZ and
NPY files for data storage, given their simplicity and support across
most Python deep learning frameworks.

The implementations in this library are based upon the following file
format documents:
- **NPY**: The NPY file format is documented by the NumPy developers
           in [this note](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html)
- **NPZ**: While not explicitly documented, the NPZ format is a
           a PKZIP archive of NPY files, and thus is documented
           here: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT.

## Getting Started

Start by installing [CMake](https://cmake.org/) in the way appropriate for your
environment.

### Linux

Create a build directory and initialize the cmake project:

    mkdir build
    cd build
    cmake .. --preset release

You can then build and run the tests using:

    make
    ctest

### Windows

Create a build directory and initialize the cmake project:

    mkdir build
    cd build
    cmake .. --preset release

You can then build and run the tests using:

    cmake --build . --config Release
    ctest -C Release

## Sample code

Once the library has been built and installed, you can begin to use it
in your code. We have provided some
[sample programs](https://github.com/matajoh/libnpy/tree/main/samples)
(and naturally the [tests](https://github.com/matajoh/libnpy/tree/main/test)
as well) which show how to use the library, but the basic concepts are as follows.
For the purpose of this sample code we will use the built-in [tensor](src/tensor.h)
class, but you should use your own tensor class as appropriate.

```C++
#include "tensor.h"
#include "npy.h"
#include "npz.h"

...
    // create a tensor object
    std::vector<size_t> shape({32, 32, 3});
    npy::tensor<std::uint8_t> color(shape);

    // fill it with some data
    for (int row = 0; row < color.shape(0); ++row)
    {
        for (int col = 0; col < color.shape(1); ++col)
        {
            color(row, col, 0) = static_cast<std::uint8_t>(row << 3);
            color(row, col, 1) = static_cast<std::uint8_t>(col << 3);
            color(row, col, 2) = 128;
        }
    }

    // save it to disk as an NPY file
    npy::save("color.npy", color);

    // we can manually set the endianness to use
    npy::save("color.npy", color, npy::endian_t::BIG);

    // the built-in tensor class also has a save method
    color.save("color.npy");

    // we can peek at the header of the file
    npy::header_info header = npy::peek("color.npy");

    // we can load it back the same way
    color = npy::load<std::uint8_t, npy::tensor>("color.npy");

    // let's create a second tensor as well
    shape = {32, 32};
    npy::tensor<float> gray(shape);

    for (int row = 0; row < gray.shape(0); ++row)
    {
        for (int col = 0; col < gray.shape(1); ++col)
        {
            gray(row, col) = 0.21f * color(row, col, 0) +
                             0.72f * color(row, col, 1) +
                             0.07f * color(row, col, 2);
        }
    }

    // we can write them to an NPZ file
    {
        npy::onpzstream output("test.npz");
        output.write("color.npy", color);
        output.write("gray.npy", gray);
    }

    // and we can read them back out again
    {
        npy::inpzstream input("test.npz");

        // we can test to see if the archive contains a file
        if (input.contains("color.npy"))
        {
            // and peek at its header
            header = input.peek("color.npy");
        }

        color = input.read<std::uint8_t>("color.npy");
        gray = input.read<float>("gray.npy");
    }
```

The generated documentation contains more details on all of the functionality.
We hope you find that the library fulfills your needs and is easy to use, but
if you have any difficulties please create
[issues](https://github.com/matajoh/libnpy/issues) so the maintainers can make
the library even better. Thanks!
