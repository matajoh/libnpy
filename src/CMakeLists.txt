set( SOURCES
   dtype.cpp
   npy.cpp
   npz.cpp
   tensor.cpp
   zip.cpp
)

set( HEADERS
   dtype.h
   util.h
   npy.h
   npz.h
   tensor.h
   zip.h
)

add_definitions( -DLIBNPY_VERSION=${LIBNPY_VERSION} )

set( LIBNPY_INCLUDE_FOLDERS 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ZLIB_INCLUDE_DIRS}
)

include_directories( ${LIBNPY_INCLUDE_FOLDERS} )

add_library( npy STATIC ${SOURCES} ${HEADERS} )

set( LIBNPY_LINK_LIBRARIES 
  ${ZLIB_LIBRARIES}
)

target_link_libraries( npy ${LIBNPY_LINK_LIBRARIES} )

install(TARGETS npy 
  RUNTIME DESTINATION "bin"
  ARCHIVE DESTINATION "lib"
  LIBRARY DESTINATION "lib"
)
install(FILES ${HEADERS} DESTINATION "include" )

# -------------------- Walk the subdirectories --------------------
if( INCLUDE_CSHARP )
  add_subdirectory( CSharpWrapper )
endif()
