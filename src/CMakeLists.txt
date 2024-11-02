set( SOURCES
   dtype.cpp
   npy.cpp
   npz.cpp
   tensor.cpp
   zip.cpp
   miniz/miniz.cpp
)

add_definitions( -DLIBNPY_VERSION=${LIBNPY_VERSION} )

add_library( npy STATIC ${SOURCES} )
add_library( npy::npy ALIAS npy )

if (LIBNPY_SANITIZE)
  target_compile_options(npy PUBLIC -g -fsanitize=${REGOCPP_SANITIZE} -fno-omit-frame-pointer)
  target_link_libraries(npy PUBLIC -fsanitize=${REGOCPP_SANITIZE})
endif()

if ( CMAKE_COMPILER_IS_GNUCC )
  target_compile_options(npy PRIVATE "-Wall" "-Wextra")
elseif( MSVC )
  target_compile_options(npy PRIVATE "/W4")
endif()
