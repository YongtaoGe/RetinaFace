cmake_minimum_required(VERSION 3.14)
project(tvm_deploy)

set(CMAKE_CXX_STANDARD 14)

# opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "cmake source dir: ${CMAKE_SOURCE_DIR}")
#target_link_libraries(tvm_deploy LINK_PRIVATE ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/deps/tvm/osx/lib/libtvm_runtime.dylib)

# tvm
set(TVM_HOME /Users/geyongtao/Desktop/tvm)
include_directories(${TVM_HOME}/include)
include_directories(${TVM_HOME}/3rdparty/dmlc-core/include)
include_directories(${TVM_HOME}/3rdparty/dlpack/include)
link_directories(${TVM_HOME}/build)
add_executable(tvm_deploy main.cpp deps/tvm/tvm_runtime_pack.cc include/detection/anchor_generator.hpp include/engine/tensor.h include/engine/syncedmem.h src/detection/anchor_generator.cpp src/engine/tensor.cpp src/engine/syncedmem.cpp include/detection/detector.hpp src/detection/detector.cpp include/misc/time.hpp src/misc/time.cpp)

#target_link_libraries(tvm_deploy LINK_PRIVATE ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/deps/tvm/osx/lib/libtvm_runtime.dylib)
target_link_libraries(tvm_deploy LINK_PRIVATE ${OpenCV_LIBS})