set( SOURCES
   dtype.cpp
   memstream.cpp
   npy.cpp
   npz.cpp
   tensor.cpp
   zip.cpp
   miniz/miniz.cpp
)

add_definitions( -DLIBNPY_VERSION=${LIBNPY_VERSION} )

add_library( npy STATIC ${SOURCES} )
add_library( npy::npy ALIAS npy )
