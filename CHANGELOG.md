# Changelog

## [2021-02-10 - Version 1.3.0](https://github.com/matajoh/libnpy/releases/tag/v1.3.0)

New Features:
- Support for Unicode string tensors (npy type 'U')

## [2021-02-09 - Version 1.2.2](https://github.com/matajoh/libnpy/releases/tag/v1.2.2)

Improvements:
- Bug fix for a missing comma on 1d shape

## [2021-02-08 - Version 1.2.1](https://github.com/matajoh/libnpy/releases/tag/v1.2.1)

Improvements:
- Bug fix for scalar tensor reading
- Bug fix with memstream buffer size at initialization
- ".npy" will be added to tensor names in NPZ writing if not already present

## [2021-01-19 - Version 1.2.0](https://github.com/matajoh/libnpy/releases/tag/v1.2.0)

New Features:
- Easier indexing (variable argument index method + negative indexes)
- Easier access to shape

Improvements:
- Cmake upgraded to "modern" usage, i.e. you use the library by adding `npy::npy` as a link library

## [2021-01-16 - Version 1.1.1](https://github.com/matajoh/libnpy/releases/tag/v1.1.1)

Improvements:
- Minor cmake change

## [2021-01-16 - Version 1.1.0](https://github.com/matajoh/libnpy/releases/tag/v1.1.0)

New Features:
- Zip64 compatibility

Improvements:
- Can use `numpy` style lookup for tensors (i.e. dropping the `.npy` from the name)
- Added a crc32 test

## [2021-01-15 - Version 1.0.0](https://github.com/matajoh/libnpy/releases/tag/v1.0.0)

New Features:
- There is no longer a dependency on `zlib`

Improvements:
- Better packaging (NuGet packages are now produced for C++ and C#)

## [2019-04-01 - Version 0.2.0](https://github.com/matajoh/libnpy/releases/tag/v0.2.0)

Breaking changes:
- Renamed `endian` => `endian_t`
- Renamed `data_type` => `data_type_t`
- Renamed `compression_method` => `compression_method_t`

New Features:
- Cleaned up exception handling. There are now tests for exceptions being correctly thrown, and the exceptions are properly wrapped for .NET- 
- Added peeking for NPY files to get the header information, and contains/peek functionality for `inpzstream`.

Improvements:
- Removed the unnecessary copies in the compression/decompression process

## [2019-03-24 - Version 0.1.0](https://github.com/matajoh/libnpy/releases/tag/v0.1.0)

Initial Release