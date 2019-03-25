using System;
using System.Collections.Generic;
using System.IO;
using NumpyIO;

namespace Testing
{
    class TestNPZWrite
    {
        static void TestWrite(bool compressed, ref int result)
        {
            string filename = compressed ? "test_compressed.npz" : "test.npz";
            byte[] expected = File.ReadAllBytes(Path.Combine("assets", "test", filename));

            UInt8Tensor color = Test.Tensor<UInt8Tensor, byte, UInt8Buffer>(new Shape(new uint[] { 5, 5, 3 }));
            Float32Tensor depth = Test.Tensor<Float32Tensor, float, Float32Buffer>(new Shape(new uint[] { 5, 5 }));
            string path = Path.GetRandomFileName();
            NPZOutputStream stream = new NPZOutputStream(path, compressed ? CompressionMethod.DEFLATED : CompressionMethod.STORED);
            stream.Write("color.npy", color);
            stream.Write("depth.npy", depth);
            stream.Close();

            byte[] actual = File.ReadAllBytes(path);

            string tag = "c#_npz_write";
            if (compressed)
            {
                tag += "_compressed";
            }
            Test.AssertEqual<byte, byte[]>(expected, actual, ref result, tag);

            File.Delete(path);
        }

        public static int Main()
        {
            int result = Test.EXIT_SUCCESS;

            TestWrite(false, ref result);
            TestWrite(true, ref result);

            return result;
        }
    }
}