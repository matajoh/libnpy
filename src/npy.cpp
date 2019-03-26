#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <cctype>
#include <tuple>

#include "npy.h"

namespace
{
const int STATIC_HEADER_LENGTH = 10;
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

    throw std::logic_error("Boolean not found");
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
void write_npy_header(std::ostream &output,
                      const std::string &dtype,
                      bool fortran_order,
                      const std::vector<size_t> &shape)
{
    std::stringstream buff;
    buff << "{'descr': '" << dtype;
    buff << "', 'fortran_order': " << (fortran_order ? "True" : "False");
    buff << ", 'shape': (";
    for (auto dim = shape.begin(); dim < shape.end(); ++dim)
    {
        buff << *dim;
        if (dim < shape.end() - 1)
        {
            buff << ", ";
        }
    }
    buff << "), }";
    std::string dictionary = buff.str();
    auto dict_length = dictionary.size();
    auto header_length = dict_length + STATIC_HEADER_LENGTH + 1;
    if (header_length % 64 != 0)
    {
        header_length = ((header_length / 64) + 1) * 64;
        dict_length = header_length - STATIC_HEADER_LENGTH - 1;
    }

    unsigned char header[STATIC_HEADER_LENGTH] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00,
                                                  static_cast<unsigned char>(dict_length + 1),
                                                  0x00};
    output.write(reinterpret_cast<char *>(header), STATIC_HEADER_LENGTH);
    output << std::left << std::setw(dict_length) << dictionary << "\n";
}

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

header_info read_npy_header(std::istream &input)
{
    unsigned char header[STATIC_HEADER_LENGTH];
    input.read(reinterpret_cast<char *>(header), STATIC_HEADER_LENGTH);
    assert(header[0] == 0x93);
    assert(header[1] == 'N');
    assert(header[2] == 'U');
    assert(header[3] == 'M');
    assert(header[4] == 'P');
    assert(header[5] == 'Y');
    size_t dict_length = 0;
    if (header[6] == 0x01 && header[7] == 0x00)
    {
        dict_length = header[8] | (header[9] << 8);
    }
    else if (header[6] == 0x02 && header[7] == 0x00)
    {
        unsigned char extra[2];
        input.read(reinterpret_cast<char *>(extra), 2);
        dict_length = header[8] | (header[9] << 8) | (extra[0] << 16) | (extra[1] << 24);
    }

    std::vector<char> buffer(dict_length);
    input.read(buffer.data(), dict_length);
    std::string dictionary(buffer.begin(), buffer.end());
    return header_info(dictionary);
}
} // namespace npy