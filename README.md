# libnpy

[![Build Status](https://travis-ci.com/matajoh/libnpy.svg?token=mQKh8ae3m6BDSeGHqxyY&branch=master)](https://travis-ci.com/matajoh/libnpy)

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

There are two main ways to use the library: as a statically linked C++
library, and as a .NET DLL (on Windows using Visual Studio). In this
guide we will walk through how to compile the library and run the tests
on our currently supported platforms. These directions will likely
work for other platforms as well (the codebase is written to be clean,
portable C++ 11). If you have problems on your platform, please raise
it as an [issue](https://github.com/matajoh/libnpy/issues).


### Ubuntu 18.04 [gcc 7.3.0], Ubuntu 16.04 [gcc 5.4.0]

First, install all of the necessary dependencies:

    sudo apt-get install git cmake build-essential

If you want to build the documentation, you will also need:

    sudo apt-get install doxygen

You may also find that `cmake` is easier to use via the curses GUI:

    sudo apt-get install cmake-curses-gui

Once everything is in place, you can clone the repository and generate the
makefiles:

    git clone https://github.com/matajoh/libnpy.git
    mkdir libnpy/build
    cd libnpy/build
    cmake -DCMAKE_BUILD_TYPE=Debug ..

Your other build options are `Release` and `RelWithDebInfo`.

### Windows 10

On Windows, you can download and install the dependencies from the following
locations:

#### Install CMake
Download and run e.g. `v3.19/cmake-3.19.0-win64-x64.msi` from
https://cmake.org/files/.

#### Install git and Visual Studio.
Get the latest Windows git from https://git-scm.com/downloads. Download a
version of Visual Studio from https://visualstudio.microsoft.com/vs/. You
will need the C++ compiler (and C# compiler if needed).

#### Install SWIG (optional, only for C#)
Browse to http://swig.org/download.html and download the latest version of
`swigwin`. Unzip the directory and copy it to your `C:\` drive. Add (e.g.)
`C:\swigwin-4.0.2` to your PATH. CMake should then find swig automatically.

#### Download and install Doxygen (optional)
If you want to build the documentation, you should also download
[Doxygen](http://www.doxygen.nl/). 

#### Generate MSBuild
Now that everything is ready, cmake can generate the MSBuild files necessary
for the project. Run the following commands in a command prompt once you have
navigated to your desired source code folder:

    git clone https://github.com/matajoh/libnpy.git
    mkdir libnpy\build
    cd libnpy\build
    cmake ..

If building the C# library, you will also need to do the following:

    cmake --build . --target NumpyIONative
    cmake ..

The reason for the above is that SWIG autogenerates the C# files for the
interface in the first pass, after which CMake needs to scan the generated
directory to build the wrapper library.

### Build and Test
You are now able to build the test the library. Doing so is the same
regardless of your platform. First, navigate to the `build` folder you
created above. Then run the following commands:

    cmake --build . --config <CONFIG>

Where `<CONFIG>` is one of `Release|Debug|RelWithDebInfo`. This will build
the project, including the tests and (if selected) the documentation. You
can then do the following:

    ctest -C <CONFIG>

Where again you replace `<CONFIG>` as above will run all of the tests.
If you want to install the library, run:

    cmake --build . --config <CONFIG> --target INSTALL

## Sample code

Once the library has been built and installed, you can begin to use it
in your code. We have provided some
[sample programs](https://github.com/matajoh/libnpy/tree/master/samples)
(and naturally the [tests](https://github.com/matajoh/libnpy/tree/master/test)
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
    for (size_t row = 0; row < color.shape(0); ++row)
    {
        for (size_t col = 0; col < color.shape(1]; ++col)
        {
            color({row, col, 0}) = static_cast<std::uint8_t>(row << 3);
            color({row, col, 1}) = static_cast<std::uint8_t>(col << 3);
            color({row, col, 2}) = 128;
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

    for (size_t row = 0; row < gray.shape(0); ++row)
    {
        for (size_t col = 0; col < gray.shape(1); ++col)
        {
            gray({row, col}) = 0.21f * color({row, col, 0}) +
                               0.72f * color({row, col, 1}) +
                               0.07f * color({row, col, 2});
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