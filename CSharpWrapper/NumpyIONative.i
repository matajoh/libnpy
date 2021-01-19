%module NumpyIO
%{
    #include "npy/tensor.h"
    #include "npy/npy.h"
    #include "npy/npz.h"
    using namespace npy;
%}

%include "std_vector.i"
%include "std_string.i"
%include "stdint.i"
%include "arrays_csharp.i"
%include "typemaps.i"
%include "attribute.i"

%rename(DataType) data_type_t;
%typemap(csbase) data_type_t "byte";
enum class data_type_t : std::uint8_t {
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT32,
    FLOAT64
};

%rename(Endian) endian_t;
%typemap(csbase) endian_t "byte";
enum class endian_t : std::uint8_t {
    NATIVE,
    BIG,
    LITTLE
};

%rename(CompressionMethod) compression_method_t;
%typemap(csbase) compression_method_t "ushort";
enum class compression_method_t : std::uint16_t {
    STORED = 0,
    DEFLATED = 8
};

%template(UInt8Buffer) std::vector<unsigned char>;
%template(Int8Buffer) std::vector<signed char>;
%template(UInt16Buffer) std::vector<unsigned short>;
%template(Int16Buffer) std::vector<short>;
%template(UInt32Buffer) std::vector<unsigned int>;
%template(Int32Buffer) std::vector<int>;
%template(UInt64Buffer) std::vector<unsigned long long>;
%template(Int64Buffer) std::vector<long long>;
%template(Float32Buffer) std::vector<float>;
%template(Float64Buffer) std::vector<double>;

%template(Shape) std::vector<size_t>;

%rename(HeaderInfo) header_info;
struct header_info
{
    header_info(data_type_t dtype,
                endian_t endianness,
                bool fortran_order,
                const std::vector<size_t> &shape);


    %rename(DataType) dtype;
    data_type_t dtype;

    %rename(Endianness) endianness;
    endian_t endianness;

    %rename(FortranOrder) fortran_order;
    bool fortran_order;

    %rename(Shape) shape;
    std::vector<size_t> shape;
};

%exception peek(const std::string& path) %{
    try{
        $action
    } catch( std::invalid_argument& e) {
        SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "Invalid path location", e.what());
        return $null;
    } catch( std::logic_error& e) {
        SWIG_CSharpSetPendingException(SWIG_CSharpIOException, e.what());
        return $null;
    }
%}

%rename(Peek) peek;
header_info peek(const std::string& path);

template <typename T>
class tensor {
public:
    %apply unsigned char FIXED[] {const unsigned char *source};
    %apply signed char FIXED[] {const signed char *source};
    %apply unsigned short FIXED[] {const unsigned short *source};
    %apply short FIXED[] {const short *source};
    %apply unsigned int FIXED[] {const unsigned int *source};
    %apply int FIXED[] {const int *source};
    %apply unsigned long long FIXED[] {const unsigned long long *source};
    %apply long long FIXED[] {const long long *source};
    %apply float FIXED[] {const float *source};
    %apply double FIXED[] {const double *source};

    %exception tensor(const std::string& path) %{
        try{
            $action
        } catch( std::invalid_argument& e) {
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "Invalid path location", e.what());
            return $null;
        } catch( std::logic_error& e) {
            SWIG_CSharpSetPendingException(SWIG_CSharpIOException, e.what());
            return $null;
        }
    %}

    explicit tensor(const std::string& path);

    tensor(const std::vector<size_t>& shape, bool fortran_order=false);

    %exception save(const std::string& path, endian_t endian = endian_t::NATIVE) %{
        try{
            $action
        } catch (std::invalid_argument& e) {
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "Invalid path location", e.what());
            return $null;            
        } catch (std::logic_error& e) {
            SWIG_CSharpSetPendingException(SWIG_CSharpIOException, e.what());
            return $null;
        }
    %}

    %csmethodmodifiers save "public override";
    %rename(Save) save;
    void save(const std::string& path, endian_t endian = endian_t::NATIVE);

    %exception copy_from(const T* source, size_t nitems) %{
        try{
            $action
        } catch (std::invalid_argument& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "Incorrect number of items", e.what());
            return $null;
        }
    %}

    %csmethodmodifiers copy_from "public unsafe override";
    %rename(CopyFrom) copy_from;
    void copy_from(const T* source, size_t itemCount);

    %csmethodmodifiers values "protected override"
    %rename(getValues) values;
    const std::vector<T>& values() const;

    %csmethodmodifiers shape "protected override"
    %rename(getShape) shape;
    const std::vector<size_t> shape() const;

    %csmethodmodifiers fortran_order "protected override"
    %rename(getFortranOrder) fortran_order;
    bool fortran_order() const;

    %csmethodmodifiers dtype "protected override"
    %rename(getDataType) dtype;
    data_type_t dtype() const;

    %csmethodmodifiers size "protected override"
    %rename(getSize) size;
    size_t size() const;

    %exception get(const std::vector<std::int32_t>& index) const %{
        try{
            $action
        }catch(std::invalid_argument& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "incorrect index size", e.what());
            return $null;
        }catch(std::out_of_range& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentOutOfRangeException, "index out of range", e.what());
            return $null;
        }
    %}

    %csmethodmodifiers get "protected override"
    const T& get(const std::vector<std::int32_t>& index) const;

    %exception set(const std::vector<std::int32_t>& index, const T& value) const %{
        try{
            $action
        }catch(std::invalid_argument& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "incorrect index size", e.what());
            return $null;
        }catch(std::out_of_range& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentOutOfRangeException, "index out of range", e.what());
            return $null;
        }
    %}

    %csmethodmodifiers set "protected override"
    void set(const std::vector<std::int32_t>& index, const T& value);
};

%typemap(csbase) SWIGTYPE "Tensor<byte, UInt8Buffer>";
%template(UInt8Tensor) tensor<unsigned char>;
%typemap(csbase) SWIGTYPE "Tensor<sbyte, Int8Buffer>";
%template(Int8Tensor) tensor<signed char>;
%typemap(csbase) SWIGTYPE "Tensor<ushort, UInt16Buffer>";
%template(UInt16Tensor) tensor<unsigned short>;
%typemap(csbase) SWIGTYPE "Tensor<short, Int16Buffer>";
%template(Int16Tensor) tensor<short>;
%typemap(csbase) SWIGTYPE "Tensor<uint, UInt32Buffer>";
%template(UInt32Tensor) tensor<unsigned int>;
%typemap(csbase) SWIGTYPE "Tensor<int, Int32Buffer>";
%template(Int32Tensor) tensor<int>;
%typemap(csbase) SWIGTYPE "Tensor<ulong, UInt64Buffer>";
%template(UInt64Tensor) tensor<unsigned long long>;
%typemap(csbase) SWIGTYPE "Tensor<long, Int64Buffer>";
%template(Int64Tensor) tensor<long long>;
%typemap(csbase) SWIGTYPE "Tensor<float, Float32Buffer>";
%template(Float32Tensor) tensor<float>;
%typemap(csbase) SWIGTYPE "Tensor<double, Float64Buffer>";
%template(Float64Tensor) tensor<double>;

%typemap(csbase) SWIGTYPE ""

%rename(NPZOutputStream) onpzstream;
class onpzstream {
public:
    onpzstream(const std::string& path, compression_method_t compression=compression_method_t::STORED, endian_t endian=endian_t::NATIVE);

    %rename(Close) close;
    void close();

    %exception write(const std::string& filename, const tensor<T>& tensor) %{
        try{
            $action
        }catch(std::invalid_argument& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "Unsupported compression method", e.what());
            return $null;
        }catch(std::logic_error& e){
            SWIG_CSharpSetPendingException(SWIG_CSharpIOException, e.what());
            return $null;
        }
    %}

    template <typename T>
    void write(const std::string& filename, const tensor<T>& tensor);
};

%extend onpzstream {
    %template(Write) write<unsigned char>;
    %template(Write) write<signed char>;
    %template(Write) write<unsigned short>;
    %template(Write) write<short>;
    %template(Write) write<unsigned int>;
    %template(Write) write<int>;
    %template(Write) write<unsigned long long>;
    %template(Write) write<long long>;
    %template(Write) write<float>;
    %template(Write) write<double>;
};

%rename(NPZInputStream) inpzstream;
class inpzstream {
public:
    %exception inpzstream(const std::string& path) %{
        try{
            $action
        }catch(std::invalid_argument& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "File does not exist", e.what());
            return $null;
        }catch(std::logic_error& e){
            SWIG_CSharpSetPendingException(SWIG_CSharpIOException, e.what());
            return $null;
        }
    %}

    inpzstream(const std::string& path);

    %rename(Contains) contains;
    bool contains(const std::string& filename);

    %exception peek(const std::string& filename) %{
        try{
            $action
        }catch(std::invalid_argument& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "Filename does not exist in archive", e.what());
            return $null;
        }catch(std::logic_error& e){
            SWIG_CSharpSetPendingException(SWIG_CSharpIOException, e.what());
            return $null;
        }
    %}

    %rename(Peek) peek;
    header_info peek(const std::string& filename);


    %exception read(const std::string& filename) %{
        try{
            $action
        }catch(std::invalid_argument& e){
            SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentException, "Filename does not exist in archive", e.what());
            return $null;
        }catch(std::logic_error& e){
            SWIG_CSharpSetPendingException(SWIG_CSharpIOException, e.what());
            return $null;
        }
    %}

    template <typename T>
    tensor<T> read(const std::string& filename);

    %rename(Close) close;
    void close();
};

%extend inpzstream {
    %template(ReadUInt8) read<unsigned char>;
    %template(ReadInt8) read<signed char>;
    %template(ReadUInt16) read<unsigned short>;
    %template(ReadInt16) read<short>;
    %template(ReadUInt32) read<unsigned int>;
    %template(ReadInt32) read<int>;
    %template(ReadUInt64) read<unsigned long long>;
    %template(ReadInt64) read<long long>;
    %template(ReadFloat32) read<float>;
    %template(ReadFloat64) read<double>;
};
