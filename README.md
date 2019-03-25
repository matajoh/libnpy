# libnpy

`libnpy` is a multi-platform C++ library for reading and writing NPY files,
with an additional .NET interface. It was built with the intention of
making it easier for multi-language projects to use NPZ and NPY files
for data storage, given their simplicity and support across most Python
deep learning frameworks.

## Getting Started

There are two main ways to use the library: as a statically linked C++
library, and as a .NET DLL (on Windows using Visual Studio). In this
guide we will walk through how to compile the library and run the tests
on our currently supported platforms. These directions will likely
work for other platforms as well (the codebase is written to be clean,
portable C++ 11). If you have problems on your platform, please raise
it as an [issue](https://github.com/matajoh/libnpy/issues).

### Ubuntu 18.04 [gcc 7.3.0]

First, install all of the necessary dependencies:

    sudo apt-get install git cmake build-essential zlib1g-dev swig

If you want to build the documentation, you will also need:

    sudo apt-get install doxygen

You may also find that `cmake` is easier to use via the curses GUI:

    sudo apt-get install cmake-curses-gui

Once everything is in place, you can clone the repository and generate the makefiles:

    git clone https://github.com/matajoh/libnpy.git
    mkdir libnpy/build
    cd libnpy/build
    cmake -DCMAKE_BUILD_TYPE=Debug ..

Your other build options are `Release` and `RelWithDebInfo`. You can now build the library:

### Windows 10 [Visual Studio 2017]

On Windows, you can download and install the dependencies from the following locations:

#### Install CMake
Download and run e.g. `v3.11/cmake-3.11.0-win64-x64.msi` from https://cmake.org/files/.

#### Install git and Visual Studio.
Get the latest Windows git from https://git-scm.com/downloads. Download a version of Visual Studio from https://visualstudio.microsoft.com/vs/. You will need the C++ and C# compilers.

#### Build and install ZLib 
Download e.g. `zlib-1.2.11.zip` from http://zlib.net/ and extract to `C:\zlib-1.2.11\`. Then, open a command prompt aith Admin rights ([How-To](https://technet.microsoft.com/en-us/library/cc947813(v=ws.10).aspx)) to run the following commands:

    cd C:\zlib-1.2.11\
    cmake -G "Visual Studio 15 2017 Win64" ..
    cmake --build . --config Debug --target install
    cmake --build . --config Release --target install

This will build and install zlib on your system. Then, add `C:\Program Files\zlib\bin` to your PATH ([How To](https://support.microsoft.com/en-us/kb/310519)).

#### Install SWIG
Browse to http://swig.org/download.html and download the latest version of `swigwin`. Unzip the directory and copy it to your `C:\` drive. Add (e.g.) `C:\swigwin-3.0.12` to your PATH. CMake should then find swig automatically.

#### Download and install Doxygen (optional)
If you want to build the documentation, you should also download [Doxygen](http://www.doxygen.nl/). 

#### Generate MSBuild
Now that everything is ready, cmake can generate the MSBuild files necessary for the project. Run the following commands in a command prompt once you have navigated to your desired source code folder:

    git clone https://github.com/matajoh/libnpy.git
    mkdir libnpy\build
    cd libnpy\build
    cmake -G "Visual Studio 15 2017 Win64" ..
    cmake --build . --target NumpyIONative
    cmake ..

The reason for the above is that SWIG autogenerates the C# files for the interface in the first pass, after which CMake needs to scan the generated directory to build the wrapper library.

### Build and Test
You are now able to build the test the library. Doing so is the same regardless of your platform. First, navigate to the `build` folder you created above. Then run the following commands:

    cmake --build . --config <CONFIG>

Where `<CONFIG>` is one of `Release|Debug|RelWithDebInfo`. This will build the project, including the tests and (if selected) the documentation. You can then do the following:

    ctest -C <CONFIG>

Where again you replace `<CONFIG>` as above will run all of the tests. If you want to install the library, run:

    cmake --build . --config <CONFIG> --target INSTALL

If you would rather package up the library for distribution, run:

    cpack -C <CONFIG>

Which will create a distribution package similar to the ones we have produced for your platform.

## Creating a tensor

Once the library has been built and installed, you can begin to use it in your code. We have provided some [sample programs](https://github.com/matajoh/libnpy/tree/master/samples) (and naturally the [tests](https://github.com/matajoh/libnpy/tree/master/test) as well) which show how to use the library, but the basic concepts are as follows. For the purpose of this sample code we will use the built-in [tensor](src/tensor.h) class, but you should use your own tensor class as appropriate.

```C++
#include "tensor.h"
#include "npy.h"
#include "npz.h"

...

    // create a tensor object
    std::vector<size_t> shape({32, 32, 3});
    npy::tensor<std::uint8_t> color(shape);

    // fill it with some data
    for(size_t row=0; row < color.shape()[0]; ++row){
        for(size_t col=0; col < color.shape()[1]; ++col){
            color({row, col, 0}) = static_cast<std::uint8_t>(row << 3);
            color({row, col, 1}) = static_cast<std::uint8_t>(col << 3);
            color({row, col, 2}) = 128;
        }
    }

    // save it to disk as an NPY file
    npy::save("color.npy", color);

    // we can manually set the endianness to use
    npy::save("color.npy", color, npy::endian::BIG);

    // the built-in tensor class also has a save method
    color.save("color.npy");

    // we can load it back the same way
    color = npy::load<std::uint8_t, npy::tensor>("color.npy");

    // let's create a second tensor as well
    shape = {32, 32};
    npy::tensor<float> gray(shape);

    for(size_t row=0; row<gray.shape()[0]; ++row){
        for(size_t col=0; col<gray.shape()[1]; ++col){
            gray({row, col}) = 0.21f*color({row, col, 0}) +
                               0.72f*color({row, col, 1}) +
                               0.07f*color({row, col, 2});
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
        color = input.read<std::uint8_t>("color.npy");
        gray = input.read<float>("gray.npy");
    }
```

The generated documentation contains more details on all of the functionality. We hope you find that the library fulfills your needs and is easy to use, but if you have any difficulties please create [issues](https://github.com/matajoh/libnpy/issues) so the maintainers can make the library even better. Thanks!