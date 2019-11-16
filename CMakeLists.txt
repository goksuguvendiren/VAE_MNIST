cmake_minimum_required(VERSION 3.8)
project(mnist)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(mnist main.cpp src/vae.cpp)

target_include_directories(mnist PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(mnist PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(mnist PUBLIC ${OpenCV_LIBS})
target_link_libraries(mnist PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(mnist PUBLIC ${PYTHON_LIBRARIES})
