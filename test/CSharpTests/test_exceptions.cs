using System;
using System.IO;
using NumpyIO;

namespace Testing
{
    class TestExceptions
    {
        static UInt8Tensor TENSOR = new UInt8Tensor(new Shape(new uint[] { 5, 2, 5 }));
        static string TEMP_NPZ = "temp.npz";

        static void PeekInvalidPath()
        {
            NumpyIO.NumpyIO.Peek(Path.Combine("does_not_exist", "bad.npy"));
        }

        static void SaveInvalidPath()
        {
            TENSOR.Save(Path.Combine("does_not_exist", "bad.npy"));
        }

        static void LoadInvalidPath()
        {
            var tensor = new UInt8Tensor(Path.Combine("does_not_exist", "bad.npy"));
        }

        static void NPZInputStreamInvalidPath()
        {
            var stream = new NPZInputStream(Path.Combine("does_not_exist", "bad.npz"));
        }

        static void NPZOutputStreamCompression()
        {
            CompressionMethod method = (CompressionMethod)99;
            using(var stream = new NPZOutputStream(TEMP_NPZ, method))
            {
                stream.Write("error.npy", TENSOR);   
            }
        }

        static void NPZInputStreamReadInvalidFilename()
        {
            using(var stream = new NPZInputStream(Test.AssetPath("test.npz")))
            {
                var tensor = stream.ReadUInt8("not_there.npy");
            }
        }

        static void NPZInputStreamPeekInvalidFilename()
        {
            using(var stream = new NPZInputStream(Test.AssetPath("test.npz")))
            {
                var header = stream.Peek("not_there.npy");
            }
        }        

        static void TensorCopyFrom()
        {
            byte[] buffer = new byte[10];
            TENSOR.CopyFrom(buffer, 10);
        }

        static void TensorIndexSize()
        {
            byte value = TENSOR[0, 0];
        }

        static void TensorIndexRange()
        {
            byte value = TENSOR[2, 3, 3];
        }

        static void NPZOutputStreamClosed()
        {
            using(var stream = new NPZOutputStream(TEMP_NPZ))
            {
                stream.Close();
                stream.Write("error.npy", TENSOR);
            }
        }

        static void NPZInputStreamInvalidFile()
        {
            var stream = new NPZInputStream(Test.AssetPath("uint8.npy"));
        }

        public static int Main(string[] args)
        {
            int result = Test.EXIT_SUCCESS;

            Test.AssertThrows<ArgumentException>(PeekInvalidPath, ref result, "PeekInvalidPath");
            Test.AssertThrows<ArgumentException>(SaveInvalidPath, ref result, "SaveInvalidPath");
            Test.AssertThrows<ArgumentException>(LoadInvalidPath, ref result, "LoadInvalidPath");
            Test.AssertThrows<ArgumentException>(NPZInputStreamInvalidPath, ref result, "NPZInputStreamInvalidPath");
            Test.AssertThrows<ArgumentException>(NPZInputStreamReadInvalidFilename, ref result, "NPZInputStreamReadInvalidFilename");
            Test.AssertThrows<ArgumentException>(NPZInputStreamPeekInvalidFilename, ref result, "NPZInputStreamPeekInvalidFilename");
            Test.AssertThrows<ArgumentException>(NPZOutputStreamCompression, ref result, "NPZOutputStreamCompression");
            Test.AssertThrows<ArgumentException>(TensorCopyFrom, ref result, "TensorCopyFrom");
            Test.AssertThrows<ArgumentException>(TensorIndexSize, ref result, "TensorIndexSize");

            Test.AssertThrows<ArgumentOutOfRangeException>(TensorIndexRange, ref result, "TensorIndexRange");

            Test.AssertThrows<IOException>(NPZInputStreamInvalidFile, ref result, "NPZInputStreamInvalidFile");
            Test.AssertThrows<IOException>(NPZOutputStreamClosed, ref result, "NPZOutputStreamClosed");
            
            File.Delete(TEMP_NPZ);

            return result;
        }
    }
}