#include <fstream>
#include <sstream>
#include <cstdint>
#include <iostream>
#include <cctype>
#include <tuple>

#include "npy.h"

namespace
{
const int BUFFER_SIZE = 64 * 1024;
char BUFFER[BUFFER_SIZE];

void read(std::istream &input, char expected)
{
    char actual;
    input.get(actual);
    assert(actual == expected);
}

void read(std::istream &input, const std::string &expected)
{
    input.read(BUFFER, expected.length());
    std::string actual(BUFFER, BUFFER + expected.length());
    assert(actual == expected);
}

void skip_whitespace(std::istream &input)
{
    char skip;
    while (std::isspace(input.peek()))
    {
        input.get(skip);
    }
}

std::string read_to(std::istream &input, char delim)
{
    input.get(BUFFER, BUFFER_SIZE, delim);
    auto length = input.gcount();
    assert(length < BUFFER_SIZE);

    return std::string(BUFFER, BUFFER + length);
}

std::string read_string(std::istream &input)
{
    read(input, '\'');
    std::string token = read_to(input, '\'');
    read(input, '\'');
    return token;
}

bool read_bool(std::istream &input)
{
    if (input.peek() == 'T')
    {
        read(input, "True");
        return true;
    }
    else if (input.peek() == 'F')
    {
        read(input, "False");
        return false;
    }

    throw std::logic_error("Dictionary value is not a boolean");
}

std::vector<size_t> read_shape(std::istream &input)
{
    read(input, '(');
    std::stringstream tuple(read_to(input, ')'));
    read(input, ')');

    std::vector<size_t> shape;
    size_t size;

    while (tuple >> size)
    {
        shape.push_back(size);
        if (tuple.peek() == ',')
        {
            read(tuple, ',');
            skip_whitespace(tuple);
        }
    }

    return shape;
}
} // namespace

namespace npy
{
header_info::header_info(const std::string &dictionary)
{
    std::istringstream input(dictionary);
    read(input, '{');
    while (input.peek() != '}')
    {
        skip_whitespace(input);
        std::string key = read_string(input);
        skip_whitespace(input);
        read(input, ':');
        skip_whitespace(input);
        if (key == "descr")
        {
            std::tie(this->dtype, this->endianness) = from_dtype(read_string(input));
        }
        else if (key == "fortran_order")
        {
            this->fortran_order = read_bool(input);
        }
        else if (key == "shape")
        {
            this->shape = read_shape(input);
        }
        else
        {
            throw std::logic_error("Unsupported key: " + key);
        }

        read(input, ',');
        skip_whitespace(input);
    }

    read(input, '}');
}

header_info::header_info(data_type_t dtype, npy::endian_t endianness, bool fortran_order, const std::vector<size_t> &shape)
{
    this->dtype = dtype;
    this->endianness = endianness;
    this->fortran_order = fortran_order;
    this->shape = shape;
}

header_info peek(const std::string &path)
{
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input.is_open())
    {
        throw std::invalid_argument("path");
    }

    return peek(input);
}

} // namespace npy