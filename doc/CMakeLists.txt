set( README_FILES
  ../README.md
  ../CHANGELOG.md
  ../LICENSE
)

# copy these files into the root of the distribution zip
install( FILES ${README_FILES} DESTINATION "." )

# build the documentation from the C++ comments
configure_file( 
  Doxyfile.in 
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY 
)

add_custom_target( npy_doc
  ALL
  ${DOXYGEN_EXECUTABLE} 
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen" VERBATIM
)
