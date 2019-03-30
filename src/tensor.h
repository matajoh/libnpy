// ----------------------------------------------------------------------------
//
// tensor.h -- default tensor class for use with the library.
//
// Copyright (C) 2019 Matthew Johnson
//
// For conditions of distribution and use, see copyright notice in LICENSE
//
// ----------------------------------------------------------------------------

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <vector>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "core.h"
#include "npy.h"

namespace npy
{
/** The default tensor class. This class can be used as a data exchange format
 *  for the library, but the methods and classes will also work with your own
 *  tensor implementation. The library methods require the following methods to
 *  be present in a tensor type:
 *  - \link data \endlink
 *  - \link shape \endlink
 *  - \link size \endlink
 *  - \link dtype \endlink
 *  - \link fortran_order \endlink
 * 
 *  As long as these are present and have the same semantics, the library should
 *  handle them in the same was as this implementation. Only certain type of
 *  tensor objects are natively supported (see \link npy::data_type_t \endlink).
 */
template <typename T>
class tensor
{
  public:
    /** Constructor.
     *  \param path the path to an NPY file on the disk
     */
    explicit tensor(const std::string &path) : tensor(npy::load<T, npy::tensor>(path)) {}

    /** Constructor. This will allocate a data buffer of the appropriate size.
     *  \param shape the shape of the tensor
     *  \param fortran_order whether the data is stored in FORTRAN, or column major, order
     */
    tensor(const std::vector<size_t> &shape,
           bool fortran_order = false) : m_shape(shape),
                                         m_ravel_strides(tensor<T>::get_ravel_strides(shape, fortran_order)),
                                         m_fortran_order(fortran_order),
                                         m_dtype(tensor<T>::get_dtype()),
                                         m_values(tensor<T>::get_size(shape))
    {
    }

    /** Copy constructor. */
    tensor(const tensor<T> &other) : m_shape(other.m_shape),
                                     m_ravel_strides(other.m_ravel_strides),
                                     m_fortran_order(other.m_fortran_order),
                                     m_dtype(other.m_dtype),
                                     m_values(other.m_values)
    {
    }

    /** Move constructor. */
    tensor(tensor<T> &&other) : m_shape(std::move(other.m_shape)),
                                m_ravel_strides(std::move(other.m_ravel_strides)),
                                m_fortran_order(other.m_fortran_order),
                                m_dtype(other.m_dtype),
                                m_values(std::move(other.m_values))
    {
    }

    /** Index function.
     *  \param multi_index the index into the tensor.
     *  \return the value at the provided index
     */
    const T &operator()(const std::vector<size_t> &multi_index) const
    {
        return this->m_values[this->ravel(multi_index)];
    }

    /** Index function.
     *  \param multi_index the index into the tensor
     *  \return a reference to the value at the index
     */
    T &operator()(const std::vector<size_t> &multi_index)
    {
        return this->m_values[this->ravel(multi_index)];
    }

    /** Sets the value at the provided index.
     *  \param multi_index an index into the tensor
     *  \param value the value to set
     */
    void set(const std::vector<size_t> &multi_index, const T &value)
    {
        this->m_values[this->ravel(multi_index)] = value;
    }

    /** Gets the value at the provided index.
     *  \param multi_index the index into the tensor
     *  \return the value at the provided index
     */
    const T &get(const std::vector<size_t> &multi_index) const
    {
        return this->m_values[this->ravel(multi_index)];
    }

    /** The data type of the tensor. */
    const data_type_t dtype() const
    {
        return this->m_dtype;
    }

    /** The underlying values buffer. */
    const std::vector<T> &values() const
    {
        return this->m_values;
    }

    /** Copy values from the source to this tensor.
     *  \param source pointer to the start of the source buffer
     *  \param nitems the number of items to copy. Should be equal to \link size \endlink.
     */
    void copy_from(const T *source, size_t nitems)
    {
        if (nitems != this->size())
        {
            throw std::invalid_argument("nitems");
        }

        std::copy(source, source + nitems, this->m_values.begin());
    }

    /** Copy values from the provided vector.
     *  \param source the source vector. Should have the same size as \link values \endlink.
     */
    void copy_from(const std::vector<T> &source)
    {
        if (source.size() != this->size())
        {
            throw std::invalid_argument("source.size");
        }

        std::copy(source.begin(), source.end(), this->m_values.begin());
    }

    /** Move values from the provided vector.
     *  \param source the source vector. Should have the same size as \link values \endlink.
     */
    void move_from(std::vector<T> &&source)
    {
        if (source.size() != this->size)
        {
            throw std::invalid_argument("source.size");
        }

        this->m_values = std::move(source);
    }

    /** A pointer to the start of the underlying values buffer. */
    T *data()
    {
        return this->m_values.data();
    }

    /** A pointer to the start of the underlying values buffer. */
    const T *data() const
    {
        return this->m_values.data();
    }

    /** The number of elements in the tensor. */
    size_t size() const
    {
        return this->m_values.size();
    }

    /** The shape of the vector. Each element is the size of the 
         *  corresponding dimension. */
    const std::vector<size_t> &shape() const
    {
        return this->m_shape;
    }

    /** Whether the tensor data is stored in FORTRAN, or column-major, order. */
    bool fortran_order() const
    {
        return this->m_fortran_order;
    }

    /** Copy assignment operator. */
    tensor<T> &operator=(const tensor<T> &other)
    {
        this->m_shape = other.m_shape;
        this->m_ravel_strides = other.m_ravel_strides;
        this->m_fortran_order = other.m_fortran_order;
        this->m_dtype = other.m_dtype;
        this->m_values = other.m_values;
        return *this;
    }

    /** Move assignment operator. */
    tensor<T> &operator=(tensor<T> &&other)
    {
        this->m_shape = std::move(other.m_shape);
        this->m_ravel_strides = std::move(other.m_ravel_strides);
        this->m_fortran_order = other.m_fortran_order;
        this->m_dtype = other.m_dtype;
        this->m_values = std::move(other.m_values);
        return *this;
    }

    /** Save this tensor to the provided location on disk.
     *  \param path a valid location on disk
     *  \param endianness the endianness to use in writing the tensor
     */
    void save(const std::string &path, endian_t endianness = npy::endian_t::NATIVE)
    {
        npy::save(path, *this, endianness);
    }

    /** Ravels a multi-index into a single value indexing the buffer.
     *  \tparam INDEX_IT the index iterator class
     *  \tparam SHAPE_IT the shape iterator class
     *  \param index the multi-index iterator
     *  \param shape the shape iterator
     *  \return the single value in the buffer corresponding to the multi-index
     */
    template <class INDEX_IT, class SHAPE_IT>
    size_t ravel(INDEX_IT index, SHAPE_IT shape) const
    {
        std::size_t ravel = 0;
        for (auto stride = this->m_ravel_strides.begin();
             stride < this->m_ravel_strides.end();
             ++index, ++shape, ++stride)
        {
            if (*index >= *shape)
            {
                throw std::out_of_range("multi_index");
            }

            ravel += *index * *stride;
        }

        return ravel;
    }

    /** Ravels a multi-index into a single value indexing the buffer.
     *  \param multi_index the multi-index value
     *  \return the single value in the buffer corresponding to the multi-index
     */
    size_t ravel(const std::vector<size_t> &multi_index) const
    {
        if (multi_index.size() != this->m_shape.size())
        {
            throw std::invalid_argument("multi_index");
        }

        if (this->m_fortran_order)
        {
            return this->ravel(multi_index.rbegin(), this->m_shape.rbegin());
        }

        return this->ravel(multi_index.begin(), this->m_shape.begin());
    }

  private:
    std::vector<size_t> m_shape;
    std::vector<size_t> m_ravel_strides;
    bool m_fortran_order;
    data_type_t m_dtype;
    std::vector<T> m_values;

    /** Returns the data type for this tensor. */
    static data_type_t get_dtype();

    /** Gets the size of a tensor given its shape */
    static size_t get_size(const std::vector<size_t> &shape)
    {
        size_t size = 1;
        for (auto &dim : shape)
        {
            size *= dim;
        }

        return size;
    }

    /** Gets the strides for ravelling */
    static std::vector<size_t> get_ravel_strides(const std::vector<size_t> &shape, bool fortran_order)
    {
        std::vector<size_t> ravel_strides(shape.size());
        size_t stride = 1;
        auto ravel = ravel_strides.rbegin();
        if (fortran_order)
        {
            for (auto max_index = shape.begin(); max_index < shape.end(); ++max_index, ++ravel)
            {
                *ravel = stride;
                stride *= *max_index;
            }
        }
        else
        {
            for (auto max_index = shape.rbegin(); max_index < shape.rend(); ++max_index, ++ravel)
            {
                *ravel = stride;
                stride *= *max_index;
            }
        }

        return ravel_strides;
    }
};
} // namespace npy

#endif