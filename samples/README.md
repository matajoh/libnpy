# Sample Code

The code here provides examples of how to use the library and the various functions it supports.

## Getting Started

To build the sample code, start by using the dependency install guide for your platform below.

### Ubuntu 18.04 [gcc 7.3.0], Ubuntu 16.04 [gcc 5.4.0]

First, install all of the necessary dependencies:

    sudo apt-get install git cmake build-essential python3 python3-pip

You may find that `cmake` is easier to use via the curses GUI:

    sudo apt-get install cmake-curses-gui

Then, install the python dependencies:

    pip3 install -r requirements.txt

Once everything is in place, you can clone the repository and generate the
makefiles:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug ..

Your other build options are `Release` and `RelWithDebInfo`.

### Windows 10 [Visual Studio 2017]

On Windows, you can download and install the dependencies from the following
locations:

#### Install CMake
Download and run e.g. `v3.11/cmake-3.11.0-win64-x64.msi` from
https://cmake.org/files/.

#### Install git and Visual Studio.
Get the latest Windows git from https://git-scm.com/downloads. Download a
version of Visual Studio from https://visualstudio.microsoft.com/vs/. You
will need the C++ and C# compilers.

#### Download and install Python

Download and run e.g. `Python 3.6.8` from https://www.python.org/downloads/windows/.

#### Generate MSBuild
Now that everything is ready, cmake can generate the MSBuild files necessary
for the project. Run the following commands in a command prompt once you have
navigated to your desired source code folder:

    mkdir build
    cd build
    cmake -G "Visual Studio 15 2017 Win64" ..

### Build and Run
You are now able to build the test the library. Doing so is the same
regardless of your platform. First, navigate to the `build` folder you
created above. Then run the following commands:

    cmake --build . --config <CONFIG>

Where `<CONFIG>` is one of `Release|Debug|RelWithDebInfo`. This will build
the project, including the tests and (if selected) the documentation. You
can then browse to the output directory and do the following:

```
./images
python display.py
```

You should see the following screen:

![Expected output of display.py](display.png)

Which indicates the sample has run correctly.

If you have built the .NET sample, then you can simply substitute `images_net` for `images` above and the same screen should appear.
