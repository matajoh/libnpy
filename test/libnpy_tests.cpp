#include <iostream>
#include <map>
#include <functional>
#include <cmath>
#include <fstream>
#include <sstream>

#include "libnpy_tests.h"

namespace {
    inline char sep()
    {
    #if defined(_WIN32) || defined(WIN32)
        return '\\';
    #else
        return '/';
    #endif
    }

}

namespace test
{

std::string path_join(const std::vector<std::string>& parts)
{
    std::stringstream result;
    result << parts.front();
    
    for(auto it = parts.begin() + 1; it < parts.end(); ++it)
    {
        result << sep() << *it;
    }

    return result.str();
}    


void assert_near(float expected, float actual, int& result, const std::string& tag)
{
    if(std::fabs(expected - actual) > 0.001f)
    {
        result = EXIT_FAILURE;
        std::cout << tag << " is incorrect: " << actual << " != " << expected << std::endl;
    }    
}

void assert_near(double expected, double actual, int& result, const std::string& tag)
{
    if(std::fabs(expected - actual) > 0.001)
    {
        result = EXIT_FAILURE;
        std::cout << tag << " is incorrect: " << actual << " != " << expected << std::endl;
    }    
}

void assert_near(const std::vector<double>& expected, const std::vector<double>& actual, int& result, const std::string& tag)
{
    assert_equal(expected.size(), actual.size(), result, tag + " size");
    if(result == EXIT_SUCCESS)
    {
        for(std::size_t i = 0; i < expected.size(); ++i)
        {
            assert_near(expected[i], actual[i], result, tag + "[" + std::to_string(i) + "]");
            if(result == EXIT_FAILURE)
            {
                break;
            }
        }
    }
}

 std::string read_file(const std::string& path){
    std::ifstream file(path, std::ios::in|std::ios::binary);
    if (!file.is_open())
    {
        throw new std::logic_error("failed to open asset file");
    }

    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();       
}

std::string read_asset(const std::string& filename){
    std::string path = path_join({"assets", "test", filename});
    return read_file(path);    
}

}

typedef std::function<int()> TestFunction;

int main(int argc, char ** argv)
{
    std::map<std::string, TestFunction> tests;

    tests["npy_write"] = test_npy_write;
    tests["npy_read"] = test_npy_read;
    tests["npz_write"] = test_npz_write;
    tests["npz_read"] = test_npz_read;
    tests["tensor"] = test_tensor;

    if(argc == 2)
    {
        std::string test(argv[1]);
        if(tests.count(test))
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
        for(auto& test : tests)
        {
            std::cout << "Running " << test.first << "..." << std::endl;
            if(test.second())
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