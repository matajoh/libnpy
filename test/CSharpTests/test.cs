using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Linq;
using NumpyIO;

namespace Testing
{
    static class Test
    {
        public const int EXIT_SUCCESS = 0;
        public const int EXIT_FAILURE = 1;

        public static void AssertEqual<T>(T expected, T actual, ref int result, string tag)
        {
            if (!actual.Equals(expected))
            {
                Console.WriteLine("{0} is incorrect: {1} != {2}", tag, actual, expected);
                result = EXIT_FAILURE;
            }
        }

        public static void AssertEqual<T, B>(B expected, B actual, ref int result, string tag) where B : IList<T>
        {
            AssertEqual(expected.Count, actual.Count, ref result, tag + " Count");
            if (result == EXIT_SUCCESS)
            {
                for (int i = 0; i < actual.Count; ++i)
                {
                    AssertEqual(expected[i], actual[i], ref result, tag + "[" + i + "]");
                    if (result == EXIT_FAILURE)
                    {
                        break;
                    }
                }
            }
        }

        public static void AssertEqual<T, B>(Tensor<T, B> expected, Tensor<T, B> actual, ref int result, string tag) where B : IList<T>
        {
            AssertEqual(expected.DataType, actual.DataType, ref result, tag + " DataType");
            AssertEqual(expected.FortranOrder, actual.FortranOrder, ref result, tag + " FortranOrder");
            AssertEqual<uint, List<uint>>(expected.Shape.ToList(), actual.Shape.ToList(), ref result, tag + " Shape");
            AssertEqual<T, IList<T>>(expected.Values, actual.Values, ref result, tag);
        }

        public static void AssertEqual(HeaderInfo expected, HeaderInfo actual, ref int result, string tag)
        {
            AssertEqual(expected.DataType, actual.DataType, ref result, tag + " DataType");
            AssertEqual(expected.Endianness, actual.Endianness, ref result, tag + " Endianness");
            AssertEqual(expected.FortranOrder, actual.FortranOrder, ref result, tag + " FortranOrder");
            AssertEqual(expected.Shape, actual.Shape, ref result, tag + " Shape");
        }

        public static void AssertThrows<E>(Action action, ref int result, string tag) where E:Exception
        {
            try
            {
                action();
                result = EXIT_FAILURE;
                Console.WriteLine("{0} did not throw an exception", tag);
            }
            catch (E)
            {                
            }
            catch(Exception e)
            {
                result = EXIT_FAILURE;
                Console.WriteLine("{0} threw an unexpected exception: {1}", tag, e);
            }
        }

        public static T Tensor<T, D, B>(Shape shape)
            where B : IList<D>
            where T : Tensor<D, B>
        {
            T tensor = (T)Activator.CreateInstance(typeof(T), new object[] { shape });
            for (int i = 0; i < tensor.Size; ++i)
            {
                tensor.Values[i] = (D)Convert.ChangeType(i, typeof(D));
            }

            return tensor;
        }

        public static string AssetPath(string filename)
        {
            return Path.Combine("assets", "test", filename);
        }
    }
}