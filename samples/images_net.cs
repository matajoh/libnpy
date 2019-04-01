using System;
using NumpyIO;

public unsafe class Sample
{
    public static void Main(string[] args)
    {
        // create a tensor object
        Shape shape = new Shape(new uint[] { 32, 32, 3 });
        UInt8Tensor color = new UInt8Tensor(shape);

        // fill it with some data.
        for (uint row = 0; row < color.Shape[0]; ++row)
        {
            for (uint col = 0; col < color.Shape[1]; ++col)
            {
                color[row, col, 0] = (byte)(row << 3);
                color[row, col, 1] = (byte)(col << 3);
                color[row, col, 2] = 128;
            }
        }

        // save it to disk as an NPY file
        color.Save("color.npy");

        // we can manually set the endianness to use
        color.Save("color.npy", Endian.BIG);

        // we can peek at the header of a file
        HeaderInfo header = NumpyIO.NumpyIO.Peek("color.npy");

        // we can load it using the path constructor
        color = new UInt8Tensor("color.npy");

        // let's create a second tensor as well
        shape = new Shape(new uint[] { 32, 32 });
        Float32Tensor gray = new Float32Tensor(shape);
        for (uint row = 0; row < gray.Shape[0]; ++row)
        {
            for (uint col = 0; col < gray.Shape[1]; ++col)
            {
                gray[row, col] = 0.21f * color[row, col, 0] +
                                 0.72f * color[row, col, 1] +
                                 0.07f * color[row, col, 2];
            }
        }

        // we can write them to an NPZ file
        NPZOutputStream output = new NPZOutputStream("test.npz");
        output.Write("color.npy", color);
        output.Write("gray.npy", gray);
        output.Close();

        // and we can read them back out again
        NPZInputStream input = new NPZInputStream("test.npz");

        // we can check of an archive contains a file
        if (input.Contains("color.npy"))
        {
            // and peek at its header
            header = input.Peek("color.npy");
        }

        color = input.ReadUInt8("color.npy");
        gray = input.ReadFloat32("gray.npy");
        input.Close();
    }
}