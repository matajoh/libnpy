#include <cstdint>

#include "tensor.h"
#include "npy.h"
#include "npz.h"

int main()
{
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

    return 0;
}