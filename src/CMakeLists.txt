set( SOURCES
   dtype.cpp
   npy.cpp
   npz.cpp
   tensor.cpp
   zip.cpp
)

if(NOT LIBNPY_USE_SYSTEM_MINIZ)
  list(APPEND SOURCES miniz/miniz.cpp)
endif()

add_definitions( -DLIBNPY_VERSION=${LIBNPY_VERSION} )

add_library( npy STATIC ${SOURCES} )
add_library( npy::npy ALIAS npy )

if(LIBNPY_USE_SYSTEM_MINIZ)
  find_package(miniz CONFIG REQUIRED)
  target_link_libraries(npy PRIVATE miniz::miniz)
  target_compile_definitions(npy PRIVATE LIBNPY_USE_SYSTEM_MINIZ)
endif()

if (LIBNPY_SANITIZE)
  target_compile_options(npy PUBLIC -g -fsanitize=${LIBNPY_SANITIZE} -fno-omit-frame-pointer)
  target_link_libraries(npy PUBLIC -fsanitize=${LIBNPY_SANITIZE})
endif()

if ( CMAKE_COMPILER_IS_GNUCC )
  target_compile_options(npy PRIVATE "-Wall" "-Wextra")
elseif( MSVC )
  target_compile_options(npy PRIVATE "/W4")
endif()
