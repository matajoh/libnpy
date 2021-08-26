#include "npy/core.h"

#include <iostream>

namespace {
    const int BUFFER_SIZE = 64 * 1024;
}

namespace npy
{
membuf::membuf() : membuf(BUFFER_SIZE)
{
    seekpos(0);
}

membuf::membuf(size_t n)
{
    m_buffer.reserve(n);
    seekpos(0);
}

membuf::membuf(const std::vector<std::uint8_t> &buffer) : m_buffer(buffer)
{
    seekpos(0);
}

membuf::membuf(std::vector<std::uint8_t> &&buffer) : m_buffer(std::move(buffer))
{
    seekpos(0);
}

membuf *membuf::setbuf(std::uint8_t *s, std::streamsize n)
{
    m_buffer = std::vector<std::uint8_t>(s, s + n);
    m_posg = m_buffer.begin();
    m_posp = m_buffer.begin();
    return this;
}

membuf::pos_type membuf::seekoff(membuf::off_type off, std::ios_base::seekdir way, std::ios_base::openmode which)
{
    membuf::pos_type result(membuf::off_type(-1));
    if (which & std::ios_base::in)
    {
        switch (way)
        {
        case std::ios_base::beg:
            m_posg = m_buffer.begin() + off;
            break;

        case std::ios_base::end:
            m_posg = m_buffer.end() + off;
            break;

        case std::ios_base::cur:
            m_posg += off;
            break;
        
        default:
            std::cerr << "Unsupported seek direction." << std::endl;
            break;
        }

        result = static_cast<membuf::pos_type>(m_posg - m_buffer.begin());
    }

    if (which & std::ios::out)
    {
        switch (way)
        {
        case std::ios_base::beg:
            m_posp = m_buffer.begin() + off;
            break;

        case std::ios_base::end:
            m_posp = m_buffer.end() + off;
            break;

        case std::ios_base::cur:
            m_posp += off;
            break;

        default:
            std::cerr << "Unsupported seek direction." << std::endl;
            break;
        }

        result = static_cast<membuf::pos_type>(m_posp - m_buffer.begin());
    }

    return result;
}

membuf::pos_type membuf::seekpos(membuf::pos_type pos, ios_base::openmode which)
{
    membuf::pos_type result(membuf::off_type(-1));
    if (which & std::ios_base::in)
    {
        m_posg = m_buffer.begin() + pos;
        result = static_cast<membuf::pos_type>(m_posg - m_buffer.begin());
    }

    if (which & std::ios::out)
    {
        m_posp = m_buffer.begin() + pos;
        result = static_cast<membuf::pos_type>(m_posp - m_buffer.begin());
    }

    return result;
}

std::streamsize membuf::showmanyc()
{
    return m_buffer.end() - m_posg;
}

std::streamsize membuf::xsgetn(std::uint8_t *s, std::streamsize n)
{
    std::streamsize bytes_read = showmanyc();
    bytes_read = n < bytes_read ? n : bytes_read;
    auto end = m_posg + bytes_read;
    std::copy(m_posg, end, s);
    m_posg = end;
    return bytes_read;
}

membuf::int_type membuf::underflow()
{
    int_type result = membuf::traits_type::eof();
    if (m_posg < m_buffer.end())
    {
        result = membuf::traits_type::to_int_type(*m_posg);
        ++m_posg;
    }

    return result;
}

membuf::int_type membuf::pbackfail(membuf::int_type c)
{
    if (c != membuf::traits_type::eof())
    {
        *m_posg = membuf::traits_type::to_char_type(c);
    }
    else
    {
        c = membuf::traits_type::to_int_type(*m_posg);
    }

    return c;
}

std::streamsize membuf::xsputn(const std::uint8_t *s, std::streamsize n)
{
    std::streamsize num_copy = m_buffer.end() - m_posp;
    num_copy = n < num_copy ? n : num_copy;
    std::streamsize num_insert = n - num_copy;

    std::copy(s, s + num_copy, m_posp);
    if (num_insert > 0)
    {
        auto diffg = m_posg - m_buffer.begin();
        m_buffer.insert(m_buffer.end(), s + num_copy, s + n);
        m_posp = m_buffer.end();
        m_posg = m_buffer.begin() + diffg;

    }else{
        m_posp += num_copy;
    }    

    return n;
}

membuf::int_type membuf::overflow(membuf::int_type c)
{
    if (c != membuf::traits_type::eof())
    {
        m_buffer.push_back(membuf::traits_type::to_char_type(c));
    }

    return c;
}

std::vector<std::uint8_t> &membuf::buf()
{
    return m_buffer;
}

const std::vector<std::uint8_t> &membuf::buf() const
{
    return m_buffer;
}

imemstream::imemstream(const std::vector<std::uint8_t> &buffer) : std::basic_istream<std::uint8_t>(&m_buffer),
                                                                  m_buffer(buffer)
{
}

imemstream::imemstream(std::vector<std::uint8_t> &&buffer) : std::basic_istream<std::uint8_t>(&m_buffer),
                                                             m_buffer(std::move(buffer))
{
}

std::vector<std::uint8_t> &imemstream::buf()
{
    return m_buffer.buf();
}

const std::vector<std::uint8_t> &imemstream::buf() const
{
    return m_buffer.buf();
}

omemstream::omemstream() : std::basic_ostream<std::uint8_t>(&m_buffer)
{
}

omemstream::omemstream(std::vector<std::uint8_t> &&buffer) : std::basic_ostream<std::uint8_t>(&m_buffer),
                                                             m_buffer(std::move(buffer))
{
}

omemstream::omemstream(std::streamsize capacity) : std::basic_ostream<std::uint8_t>(&m_buffer),
                                                   m_buffer(capacity)
{
}

std::vector<std::uint8_t> &omemstream::buf()
{
    return m_buffer.buf();
}

const std::vector<std::uint8_t> &omemstream::buf() const
{
    return m_buffer.buf();
}

} // namespace npy