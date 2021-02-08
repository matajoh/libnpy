#include "npy/core.h"

namespace {
    const int BUFFER_SIZE = 64 * 1024;
}

namespace npy
{
membuf::membuf() : membuf(BUFFER_SIZE)
{
    this->seekpos(0);
}

membuf::membuf(size_t n)
{
    m_buffer.reserve(BUFFER_SIZE);
    this->seekpos(0);
}

membuf::membuf(const std::vector<std::uint8_t> &buffer) : m_buffer(buffer)
{
    this->seekpos(0);
}

membuf::membuf(std::vector<std::uint8_t> &&buffer) : m_buffer(std::move(buffer))
{
    this->seekpos(0);
}

membuf *membuf::setbuf(std::uint8_t *s, std::streamsize n)
{
    this->m_buffer = std::vector<std::uint8_t>(s, s + n);
    this->m_posg = this->m_buffer.begin();
    this->m_posp = this->m_buffer.begin();
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
            this->m_posg = this->m_buffer.begin() + off;
            break;

        case std::ios_base::end:
            this->m_posg = this->m_buffer.end() + off;
            break;

        case std::ios_base::cur:
            this->m_posg += off;
            break;
        }

        result = static_cast<membuf::pos_type>(this->m_posg - this->m_buffer.begin());
    }

    if (which & std::ios::out)
    {
        switch (way)
        {
        case std::ios_base::beg:
            this->m_posp = this->m_buffer.begin() + off;
            break;

        case std::ios_base::end:
            this->m_posp = this->m_buffer.end() + off;
            break;

        case std::ios_base::cur:
            this->m_posp += off;
            break;
        }

        result = static_cast<membuf::pos_type>(this->m_posp - this->m_buffer.begin());
    }

    return result;
}

membuf::pos_type membuf::seekpos(membuf::pos_type pos, ios_base::openmode which)
{
    membuf::pos_type result(membuf::off_type(-1));
    if (which & std::ios_base::in)
    {
        this->m_posg = this->m_buffer.begin() + pos;
        result = static_cast<membuf::pos_type>(this->m_posg - this->m_buffer.begin());
    }

    if (which & std::ios::out)
    {
        this->m_posp = this->m_buffer.begin() + pos;
        result = static_cast<membuf::pos_type>(this->m_posp - this->m_buffer.begin());
    }

    return result;
}

std::streamsize membuf::showmanyc()
{
    return this->m_buffer.end() - this->m_posg;
}

std::streamsize membuf::xsgetn(std::uint8_t *s, std::streamsize n)
{
    std::streamsize bytes_read = this->showmanyc();
    bytes_read = n < bytes_read ? n : bytes_read;
    auto end = this->m_posg + bytes_read;
    std::copy(this->m_posg, end, s);
    this->m_posg = end;
    return bytes_read;
}

membuf::int_type membuf::underflow()
{
    int_type result = membuf::traits_type::eof();
    if (this->m_posg < this->m_buffer.end())
    {
        result = membuf::traits_type::to_int_type(*this->m_posg);
        ++this->m_posg;
    }

    return result;
}

membuf::int_type membuf::pbackfail(membuf::int_type c)
{
    if (c != membuf::traits_type::eof())
    {
        *this->m_posg = membuf::traits_type::to_char_type(c);
    }
    else
    {
        c = membuf::traits_type::to_int_type(*this->m_posg);
    }

    return c;
}

std::streamsize membuf::xsputn(const std::uint8_t *s, std::streamsize n)
{
    std::streamsize num_copy = this->m_buffer.end() - this->m_posp;
    num_copy = n < num_copy ? n : num_copy;
    std::streamsize num_insert = n - num_copy;

    std::copy(s, s + num_copy, this->m_posp);
    if (num_insert > 0)
    {
        auto diffg = this->m_posg - this->m_buffer.begin();
        this->m_buffer.insert(this->m_buffer.end(), s + num_copy, s + n);
        this->m_posp = this->m_buffer.end();
        this->m_posg = this->m_buffer.begin() + diffg;

    }else{
        this->m_posp += num_copy;
    }    

    return n;
}

membuf::int_type membuf::overflow(membuf::int_type c)
{
    if (c != membuf::traits_type::eof())
    {
        this->m_buffer.push_back(membuf::traits_type::to_char_type(c));
    }

    return c;
}

std::vector<std::uint8_t> &membuf::buf()
{
    return this->m_buffer;
}

const std::vector<std::uint8_t> &membuf::buf() const
{
    return this->m_buffer;
}

imemstream::imemstream(const std::vector<std::uint8_t> &buffer) : m_buffer(buffer),
                                                                  std::basic_istream<std::uint8_t>(&this->m_buffer)
{
}

imemstream::imemstream(std::vector<std::uint8_t> &&buffer) : m_buffer(std::move(buffer)),
                                                             std::basic_istream<std::uint8_t>(&this->m_buffer)
{
}

std::vector<std::uint8_t> &imemstream::buf()
{
    return this->m_buffer.buf();
}

const std::vector<std::uint8_t> &imemstream::buf() const
{
    return this->m_buffer.buf();
}

omemstream::omemstream() : std::basic_ostream<std::uint8_t>(&this->m_buffer)
{
}

omemstream::omemstream(std::vector<std::uint8_t> &&buffer) : m_buffer(std::move(buffer)),
                                                             std::basic_ostream<std::uint8_t>(&this->m_buffer)
{
}

omemstream::omemstream(std::streamsize capacity) : m_buffer(capacity),
                                                   std::basic_ostream<std::uint8_t>(&this->m_buffer)
{
}

std::vector<std::uint8_t> &omemstream::buf()
{
    return this->m_buffer.buf();
}

const std::vector<std::uint8_t> &omemstream::buf() const
{
    return this->m_buffer.buf();
}

} // namespace npy