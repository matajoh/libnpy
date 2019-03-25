using System;
using System.Collections.Generic;
using System.IO;
using NumpyIO;

namespace Testing
{
    class TestNPZRead
    {
        static void TestRead(bool compressed, ref int result)
        {
            UInt8Tensor expectedColor = Test.Tensor<UInt8Tensor, byte, UInt8Buffer>(new Shape(new uint[] { 5, 5, 3 }));
            Float32Tensor expectedDepth = Test.Tensor<Float32Tensor, float, Float32Buffer>(new Shape(new uint[] { 5, 5 }));

            string filename = compressed ? "test_compressed.npz" : "test.npz";
            NPZInputStream stream = new NPZInputStream(Path.Combine("assets", "test", filename));

            UInt8Tensor actualColor = stream.ReadUInt8("color.npy");
            Float32Tensor actualDepth = stream.ReadFloat32("depth.npy");

            string tag = "c#_npz_read";
            if (compressed)
            {
                tag += "_compressed";
            }

            Test.AssertEqual<byte, UInt8Buffer>(expectedColor, actualColor, ref result, tag + " color");
            Test.AssertEqual<float, Float32Buffer>(expectedDepth, actualDepth, ref result, tag + " depth");
        }
        public static int Main()
        {
            int result = Test.EXIT_SUCCESS;

            return result;
        }
    }
}