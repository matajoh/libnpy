add_executable( npy_images images.cpp )
target_link_libraries( npy_images npy::npy )

if( INCLUDE_CSHARP )
  add_executable( npy_images_net images_net.cs )
  target_link_libraries( npy_images_net NumpyIO ${SWIG_MODULE_NumpyIONative_REAL_NAME} )
endif()