if( WIN32 )
  set( SAMPLES_FOLDER ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG> )
else()
  set( SAMPLES_FOLDER ${CMAKE_CURRENT_BINARY_DIR} )
endif()

set( LIBNPY_INCLUDE_FOLDERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR} 
)
include_directories( ${LIBNPY_INCLUDE_FOLDERS} )

add_executable( images images.cpp )
target_link_libraries( images npy )
install( TARGETS images DESTINATION samples )

install( FILES images.cpp DESTINATION samples )
install( FILES CMakeLists.txt.in DESTINATION samples RENAME CMakeLists.txt )
install( FILES display.py DESTINATION samples )

add_custom_command( TARGET images POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/display.py ${SAMPLES_FOLDER}/display.py )

if( INCLUDE_CSHARP )
  add_executable( images_net images_net.cs )
  target_link_libraries( images_net NumpyIO ${SWIG_MODULE_NumpyIONative_REAL_NAME} )
  add_custom_command( TARGET images_net POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${LIBNPY_CSHARP_DIR}/${NUMPYIO_NATIVE} ${SAMPLES_FOLDER}/${NUMPYIO_NATIVE} )
  install( TARGETS images_net DESTINATION samples )

  install( FILES images_net.cs DESTINATION samples )
endif()