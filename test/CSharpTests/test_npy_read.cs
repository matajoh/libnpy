using System;
using System.Collections.Generic;
using System.IO;
using NumpyIO;

namespace Testing
{
    class TestNPYRead
    {
        static void TestRead<T, D, B>(ref int result, string tag)
                where B : IList<D>
                where T : Tensor<D, B>
        {
            Shape shape = new Shape(new uint[] { 5, 2, 5 });
            T expected = Test.Tensor<T, D, B>(shape);
            string path = Test.AssetPath(tag + ".npy");
            T actual = (T)Activator.CreateInstance(typeof(T), new object[] { path });
            Test.AssertEqual<D, B>(expected, actual, ref result, "c#_npy_read_" + tag);
        }
        public static int Main()
        {
            int result = Test.EXIT_SUCCESS;

            TestRead<UInt8Tensor, byte, UInt8Buffer>(ref result, "uint8");
            TestRead<Int8Tensor, sbyte, Int8Buffer>(ref result, "int8");
            TestRead<UInt16Tensor, ushort, UInt16Buffer>(ref result, "uint16");
            TestRead<Int16Tensor, short, Int16Buffer>(ref result, "int16");
            TestRead<UInt32Tensor, uint, UInt32Buffer>(ref result, "uint32");
            TestRead<Int32Tensor, int, Int32Buffer>(ref result, "int32");
            TestRead<UInt64Tensor, ulong, UInt64Buffer>(ref result, "uint64");
            TestRead<Int64Tensor, long, Int64Buffer>(ref result, "int64");

            return result;
        }
    }
}