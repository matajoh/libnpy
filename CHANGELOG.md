# Changelog

### 2019-04-01 - Version 0.2.0

Breaking changes:
- Renamed `endian` => `endian_t`
- Renamed `data_type` => `data_type_t`
- Renamed `compression_method` => `compression_method_t`

New Features:
- Cleaned up exception handling. There are now tests for exceptions being correctly thrown, and the exceptions are properly wrapped for .NET- 
- Added peeking for NPY files to get the header information, and contains/peek functionality for `inpzstream`.

Improvements:
- Removed the unnecessary copies in the compression/decompression process

### 2019-03-24 - Version 0.1.0

Initial Release