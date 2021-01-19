add_executable( images images.cpp )
target_link_libraries( images npy::npy )

if( INCLUDE_CSHARP )
  add_executable( images_net images_net.cs )
  target_link_libraries( images_net NumpyIO ${SWIG_MODULE_NumpyIONative_REAL_NAME} )
endif()