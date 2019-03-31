using System;
using System.Collections.Generic;
using System.IO;
using NumpyIO;

namespace Testing
{
    class TestNPYPeek
    {
        static void TestPeek(ref int result, string tag, DataType dtype, Endian endianness)
        {
            TestPeek(result, tag, dtype, endianness, false);
        }

        static void TestPeek(ref int result, string tag, DataType dtype, bool fortranOrder)
        {
            TestPeek(result, tag, dtype, Endian.LITTLE, fortranOrder);
        }
        static void TestPeek(ref int result, string tag, DataType dtype,
                             Endian endianness = Endian.LITTLE,
                             bool fortranOrder = false)
        {
            Shape shape = new Shape(new uint[]{5, 2, 5});
            HeaderInfo expected = new HeaderInfo(dtype, endianness, fortranOrder, shape);
            HeaderInfo actual = NumpyIO.NumpyIO.Peek(Test.AssetPath(tag + ".npy"));
            Test.AssertEqual(expected, actual, ref result, "c#_npy_peek_" + tag);
            File.Delete(path);
        }
        public static int Main()
        {
            int result = Test.EXIT_SUCCESS;

            TestPeek(ref result, "uint8", DataType.UINT8, Endian.NATIVE);
            TestPeek(ref result, "uint8_fortran", DataType.UINT8, Endian.NATIVE,true);
            TestPeek(ref result, "int8", DataType.INT8, Endian.NATIVE);
            TestPeek(ref result, "uint16", DataType.UINT16);
            TestPeek(ref result, "int16", DataType.INT16);
            TestPeek(ref result, "uint32", DataType.UINT32);
            TestPeek(ref result, "int32", DataType.INT32);
            TestPeek(ref result, "int32_big", DataType.INT32, Endian.BIG);
            TestPeek(ref result, "uint64", DataType.UINT64);
            TestPeek(ref result, "int64", DataType.INT64);
            TestPeek(ref result, "float32", DataType.FLOAT32);
            TestPeek(ref result, "float64", DataType.FLOAT64);

            return result;
        }
    }
}