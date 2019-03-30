#include <iostream>
#include <map>
#include <functional>
#include <cmath>
#include <fstream>
#include <sstream>

#include "libnpy_tests.h"

namespace
{
inline char sep()
{
#if defined(_WIN32) || defined(WIN32)
    return '\\';
#else
    return '/';
#endif
}

} // namespace

namespace test
{

std::string path_join(const std::vector<std::string> &parts)
{
    std::stringstream result;
    result << parts.front();

    for (auto it = parts.begin() + 1; it < parts.end(); ++it)
    {
        result << sep() << *it;
    }

    return result.str();
}

std::string read_file(const std::string &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        throw std::invalid_argument("path");
    }

    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

std::string asset_path(const std::string &filename)
{
    return path_join({"assets", "test", filename});
}

std::string read_asset(const std::string &filename)
{
    return read_file(asset_path(filename));
}

} // namespace test

typedef std::function<int()> TestFunction;

int main(int argc, char **argv)
{
    std::map<std::string, TestFunction> tests;

    tests["exceptions"] = test_exceptions;
    tests["memstream"] = test_memstream;
    tests["npy_peek"] = test_npy_peek;
    tests["npy_read"] = test_npy_read;
    tests["npy_write"] = test_npy_write;
    tests["npz_read"] = test_npz_read;
    tests["npz_write"] = test_npz_write;
    tests["tensor"] = test_tensor;

    if (argc == 2)
    {
        std::string test(argv[1]);
        if (tests.count(test))
        {
            return tests[test]();
        }
        else
        {
            std::cout << "Invalid test: " << test << std::endl;
            return EXIT_FAILURE;
        }
    }
    else
    {
        int result = EXIT_SUCCESS;
        for (auto &test : tests)
        {
            std::cout << "Running " << test.first << "..." << std::endl;
            if (test.second())
            {
                result = EXIT_FAILURE;
                std::cout << test.first << " failed." << std::endl;
            }
            else
            {
                std::cout << test.first << " succeeded." << std::endl;
            }
        }

        return result;
    }
}