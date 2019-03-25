using System;
using System.Collections.Generic;
using System.IO;
using NumpyIO;

namespace Testing
{
    class TestNPYWrite
    {
        static void TestWrite<T, D, B>(ref int result, string tag)
                where B : IList<D>
                where T : Tensor<D, B>
        {
            Shape shape = new Shape(new uint[] { 5, 2, 5 });
            T tensor = Test.Tensor<T, D, B>(shape);
            string path = Path.GetRandomFileName();
            tensor.Save(path);
            byte[] actual = File.ReadAllBytes(path);
            byte[] expected = File.ReadAllBytes(Path.Combine("assets", "test", tag + ".npy"));
            Test.AssertEqual<byte, byte[]>(expected, actual, ref result, "c#_npy_write_" + tag);
            File.Delete(path);
        }
        public static int Main()
        {
            int result = Test.EXIT_SUCCESS;

            TestWrite<UInt8Tensor, byte, UInt8Buffer>(ref result, "uint8");
            TestWrite<Int8Tensor, sbyte, Int8Buffer>(ref result, "int8");
            TestWrite<UInt16Tensor, ushort, UInt16Buffer>(ref result, "uint16");
            TestWrite<Int16Tensor, short, Int16Buffer>(ref result, "int16");
            TestWrite<UInt32Tensor, uint, UInt32Buffer>(ref result, "uint32");
            TestWrite<Int32Tensor, int, Int32Buffer>(ref result, "int32");
            TestWrite<UInt64Tensor, ulong, UInt64Buffer>(ref result, "uint64");
            TestWrite<Int64Tensor, long, Int64Buffer>(ref result, "int64");

            return result;
        }
    }
}