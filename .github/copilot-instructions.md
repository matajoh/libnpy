# libnpy — Copilot Agent Instructions

## What This Project Does

**libnpy** is a multi-platform C++17 static library for reading and writing NumPy array files:

- **NPY** — binary format for a single N-dimensional array. The format consists of a fixed magic header, a Python-dict metadata block (dtype, shape, Fortran order), and raw binary data.
- **NPZ** — a PKZIP archive containing one or more NPY files, optionally with DEFLATE compression.

The library is intended to let C++ projects exchange tensor data with Python deep learning frameworks that consume NumPy files.

The public CMake target is `npy::npy` (a static library).

---

## Repository Layout

```
include/npy/        Public API headers
  npy.h             Core types, free functions (save/load/peek), NPZ reader/writer classes
  tensor.h          Default npy::tensor<T> class

src/                Implementation (compiled into the static library)
  npy.cpp           NPY header parsing/writing; save/load/peek for NPY files
  npz.cpp           NPZ reader (npy::npzfilereader) and writer (npy::npzfilewriter)
  dtype.cpp         dtype string ↔ (data_type_t, endian_t) conversion tables
  tensor.cpp        npy::tensor<T> non-template helpers
  zip.cpp           Thin wrapper: npy_deflate / npy_inflate / npy_crc32
  zip.h             Internal zip wrapper header
  miniz/            Bundled miniz (single-file DEFLATE/inflate + CRC32 library)

test/               Unit and integration tests (CTest)
  libnpy_tests.cpp  Test driver / harness
  npy_read.cpp      NPY read tests
  npy_write.cpp     NPY write tests
  npy_peek.cpp      NPY peek (header-only inspection) tests
  npz_read.cpp      NPZ read tests
  npz_write.cpp     NPZ write tests
  npz_peek.cpp      NPZ peek tests
  tensor.cpp        tensor<T> unit tests
  custom_tensor.cpp Tests for user-defined tensor types
  crc32.cpp         CRC32 correctness tests
  exceptions.cpp    Error-handling / exception tests

assets/test/        Golden test fixtures (.npy and .npz files)

examples/           Standalone example programs
  custom_tensors/   Shows how to use the library with a user-defined tensor type
  images/           Image-based example

cmake/              CMake helper files (find-module, install config)
doc/                Doxygen configuration
vcpkg.json          vcpkg manifest (consumers who use vcpkg to manage deps)
ports/
  matajoh-libnpy/
    vcpkg.json      Port manifest (metadata, host deps)
    portfile.cmake  Build/install instructions for vcpkg
    usage           Usage hint shown after vcpkg install
build/              Out-of-source CMake build directory (not committed)
```

---

## Key Public Types

| Type | Header | Purpose |
|------|--------|---------|
| `npy::tensor<T>` | `tensor.h` | Default N-dimensional array. Supports row-major and Fortran (column-major) layout. |
| `npy::data_type_t` | `npy.h` | Enum of all supported element types: INT8/UINT8 … INT64/UINT64, FLOAT32/FLOAT64, COMPLEX64/COMPLEX128, BOOL, UNICODE_STRING. |
| `npy::endian_t` | `npy.h` | NATIVE / BIG / LITTLE. |
| `npy::boolean` | `npy.h` | Byte-sized bool wrapper (avoids `std::vector<bool>` bitfield issues). |
| `npy::header_info` | `npy.h` | Parsed NPY header: dtype, endianness, fortran_order, shape, max_element_length. |
| `npy::npzfilewriter` | `npy.h` | Streams NPY entries into a new NPZ file. |
| `npy::npzfilereader` | `npy.h` | Reads and inspects entries from an existing NPZ file. |

---

## Core API

```cpp
// NPY — single-array files
npy::header_info npy::peek(const std::string &path);
template<typename T, template<typename> class Tensor>
Tensor<T> npy::load(const std::string &path);
template<typename Tensor>
void npy::save(const std::string &path, const Tensor &tensor,
               npy::endian_t endian = npy::endian_t::NATIVE);

// NPZ — multi-array archives
npy::npzfilereader reader("file.npz");
bool reader.contains("name.npy");
npy::header_info reader.peek("name.npy");
Tensor reader.read<Tensor>("name.npy");

npy::npzfilewriter writer("file.npz");
writer.write("name.npy", tensor);        // no compression
writer.write("name.npy", tensor, true);  // with DEFLATE compression
```

---

## Custom Tensor Support

The library is not tied to `npy::tensor<T>`. Any class that exposes these five members will work transparently with all save/load/write/read overloads:

| Member | Signature | Semantics |
|--------|-----------|-----------|
| `data()` | `const T* data() const` | Pointer to contiguous element buffer |
| `shape()` | `const std::vector<size_t>& shape() const` | Size of each dimension |
| `size()` | `size_t size() const` | Total number of elements |
| `dtype()` | `npy::data_type_t dtype() const` | Element type tag |
| `fortran_order()` | `bool fortran_order() const` | Column-major flag |

See `examples/custom_tensors/` and `test/custom_tensor.cpp` for worked examples.

---

## How It Works

### NPY read path (`src/npy.cpp`)
1. Open file stream, read the 10-byte static header (magic `\x93NUMPY`, version bytes, header length).
2. Parse the Python-dict metadata string into a `header_info` (dtype string → `data_type_t` + `endian_t` via `dtype.cpp`, shape tuple, fortran_order flag).
3. Read the raw binary payload directly into the tensor's data buffer.
4. If the file endianness differs from the machine's native endianness, byte-swap each element.

### NPY write path
1. Build the Python-dict header string from shape, dtype string (via `npy::to_dtype`), and fortran_order.
2. Pad the header to a multiple of 64 bytes for alignment.
3. Write magic + version + header length field + padded header + raw binary data.

### NPZ read/write path (`src/npz.cpp` + `src/zip.cpp`)
- Uses the PKZIP local-file / central-directory structure directly (no external zlib dependency at link time — miniz is bundled).
- **Writing**: each `npzfilewriter::write` call serialises the NPY bytes into memory, optionally deflates them with `npy_deflate`, appends a local-file record, then on destruction writes the central directory and end-of-central-directory record.
- **Reading**: `npzfilereader` scans the central directory to build a name→offset index, then seeks to each local-file record on demand; compressed entries are inflated with `npy_inflate` before NPY parsing.
- CRC32 checksums are computed (via `npy_crc32` → miniz) and validated on read.

### dtype mapping (`src/dtype.cpp`)
Maintains two static lookup tables:
- `data_type_t` + `endian_t` → NPY dtype string (e.g. `"<f4"` for little-endian FLOAT32).
- NPY dtype string → `(data_type_t, endian_t)` pair.

---

## Build System

The project uses **CMake 3.15+** with C++17.

```
# Configure (choose a preset from CMakePresets.json)
cmake -B build --preset release

# Build
cmake --build build --config Release

# Test (requires LIBNPY_BUILD_TESTS=ON)
ctest -C Release --test-dir build
```

Key CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `LIBNPY_BUILD_TESTS` | OFF | Build the CTest test executable |
| `LIBNPY_BUILD_DOCUMENTATION` | OFF | Run Doxygen to generate API docs |
| `LIBNPY_USE_SYSTEM_MINIZ` | OFF | Use system-installed miniz instead of vendored copy |
| `LIBNPY_SANITIZE` | `""` | Pass a sanitizer name (e.g. `address`) |

The library installs CMake package config files (`npyConfig.cmake`, `npyTargets.cmake`) so downstream projects can consume it with `find_package(npy)`. Config files land in `share/npy/` (the standard vcpkg location) and headers in `include/`.

---

## vcpkg Deployment

### As a consumer (using vcpkg to pull libnpy into another project)

The root `vcpkg.json` declares the package metadata. Add `matajoh-libnpy` to your own `vcpkg.json` dependencies:

```json
{
  "dependencies": [ "matajoh-libnpy" ]
}
```

Then in CMake:

```cmake
find_package(npy CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE npy::npy)
```

### As an overlay port (before the port is in the vcpkg registry)

The `ports/matajoh-libnpy/` directory is a self-contained overlay port:

```
vcpkg install matajoh-libnpy --overlay-ports=ports
```

Or add to `CMakePresets.json` via `VCPKG_OVERLAY_PORTS`.

### Publishing to the vcpkg registry

Before submitting a PR to the upstream vcpkg registry:
1. Create a release tag `v<version>` on GitHub.
2. Compute the SHA512 of the archive: `vcpkg hash <archive.tar.gz>`.
3. Replace the placeholder `SHA512 0` in `ports/matajoh-libnpy/portfile.cmake` with the real hash.
4. Run `vcpkg install matajoh-libnpy --overlay-ports=ports` to verify.
5. Run `vcpkg x-add-version matajoh-libnpy --overlay-ports=ports` to register the version.

---

## Coding Conventions

- **Standard**: C++17 throughout.
- **Namespace**: all public symbols live in `npy::`.
- **Headers**: public API is in `include/npy/`; internal helpers (e.g. `zip.h`) stay in `src/`.
- **Formatting**: clang-format is enforced via the `libnpy_format` CMake target (uses clang-format-10/14/18 if found).
- **Error handling**: invalid files or unsupported configurations throw `std::runtime_error` (or derived types). See `test/exceptions.cpp`.
- **No external runtime dependencies**: miniz is vendored in `src/miniz/` so the built library has no link-time dependencies beyond the C++ standard library.

---

## Development Workflow and Deployment Process

### Core Principles

When working on this codebase, follow these fundamental principles:

#### 1. Move Slow to Go Fast
Make small, incremental changes that can be tested and understood in isolation. Breaking work into smaller pieces makes debugging easier and reduces the risk of introducing new issues. Don't try to fix everything at once—focus on one problem at a time.

#### 2. Fix Root Causes, Don't Patch Symptoms
When something goes wrong, investigate deeply to find the underlying issue rather than just addressing surface-level problems. Ask "why" multiple times to get to the real cause. A quick fix that doesn't address the root cause will lead to more problems later.

#### 3. Make a Plan and Get Approval First
Before making any changes to the codebase:
- Create a detailed plan describing the changes you intend to make
- Document the reasoning behind each change
- Save the plan to a file (e.g., `/memories/session/work-plan.md`) for tracking and reference
- Present the plan to the maintainer for approval
- Only proceed with implementation after explicit approval

#### 4. Establish a Baseline
Before beginning work:
- Run the full test suite and document the results in your plan file
- Record which tests pass and which fail (if any)
- Note the current state of the build system
- This prevents confusion about whether an error is pre-existing or newly introduced
- Always compare results against the baseline after making changes

#### 5. All Changes Need Review
Every change must go through a review process before being considered complete. See the review workflow below.

#### 6. You Don't Commit Code
As an AI agent, you prepare and test changes but never commit them to version control. The maintainer reviews and commits all changes. Focus on producing high-quality, well-tested code ready for human review.

---

### Review Process

All changes must follow this iterative review workflow:

#### Step 1: Initial Implementation
Complete your planned changes according to the approved plan.

#### Step 2: Spawn Review Subagent
- Invoke a subagent with a **fresh context** to review your changes
- Provide the subagent with:
  - The original requirements/plan
  - The changes made (diffs, file locations)
  - Any relevant context about the codebase
- The subagent should evaluate:
  - Correctness and completeness
  - Adherence to coding conventions
  - Potential edge cases or bugs
  - Test coverage
  - Documentation quality

#### Step 3: Address Review Comments
- Document all comments and suggestions from the review subagent
- Create a new plan addressing each comment
- Present this remediation plan to the maintainer for approval
- Only proceed after approval

#### Step 4: Implement Fixes
Make the changes to address review comments.

#### Step 5: Iterate
Return to Step 2 and repeat the review cycle until the subagent reviewer has no further suggestions or concerns.

#### Trivial Changes Exception
Simple changes may bypass the full review process, but **you do not decide what is trivial**. If you believe a change is trivial (e.g., fixing a typo, updating a comment), explicitly propose this to the maintainer and await confirmation. Examples that might qualify:
- Fixing obvious typos in comments or documentation
- Updating copyright years
- Correcting a broken external link

When in doubt, use the full review process.

---

### Workflow Summary

A typical development session follows this pattern:

1. **Understand the Request**: Gather context about what needs to be done
2. **Establish Baseline**: Run tests and document current state
3. **Create Plan**: Document proposed changes and get approval
4. **Implement**: Make the approved changes incrementally
5. **Review Loop**: Use subagent reviews iteratively until code is clean
6. **Final Handoff**: Present completed, reviewed changes to maintainer for commit

This process ensures quality, maintainability, and clear communication throughout the development lifecycle.
