using System;
using System.Collections.Generic;
using System.IO;
using NumpyIO;

namespace Testing
{
    class TestNPZPeek
    {
        static void TestPeek(bool compressed, ref int result)
        {
            HeaderInfo expectedColor = new HeaderInfo(DataType.UINT8, Endian.NATIVE, false, new Shape(new uint[] { 5, 5, 3 }));
            HeaderInfo expectedDepth = new HeaderInfo(DataType.FLOAT32, Endian.LITTLE, false, new Shape(new uint[] { 5, 5 }));

            string filename = compressed ? "test_compressed.npz" : "test.npz";
            NPZInputStream stream = new NPZInputStream(Test.AssetPath(filename));

            HeaderInfo actualColor = stream.Peek("color.npy");
            HeaderInfo actualDepth = stream.Peek("depth.npy");

            string tag = "c#_npz_read";
            if (compressed)
            {
                tag += "_compressed";
            }

            StringList keys = stream.Keys();
            Test.AssertEqual(keys[0], "color.npy", ref result, tag + " keys");
            Test.AssertEqual(keys[1], "depth.npy", ref result, tag + " keys");
            Test.AssertEqual(keys[2], "unicode.npy", ref result, tag + " keys");

            Test.AssertEqual(false, stream.Contains("not_there.npy"), ref result, tag + " contains not_there");
            Test.AssertEqual(true, stream.Contains("color.npy"), ref result, tag + " contains color");
            Test.AssertEqual(true, stream.Contains("depth.npy"), ref result, tag + " contains depth");
            Test.AssertEqual(expectedColor, actualColor, ref result, tag + " color");
            Test.AssertEqual(expectedDepth, actualDepth, ref result, tag + " depth");
        }
        public static int Main()
        {
            int result = Test.EXIT_SUCCESS;
            
            TestPeek(false, ref result);
            TestPeek(true, ref result);

            return result;
        }
    }
}