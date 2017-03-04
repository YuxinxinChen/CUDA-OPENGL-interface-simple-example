cmake_minimum_required(VERSION 2.8)
project(simple CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES simple.cu)

set(EXTRA_LIBRARIES "")

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()
if(LINUX)
    set(EXTRA_LIBRARIES m dl X11 Xxf86vm Xrandr pthread Xi Xcursor Xinerama)
endif()

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${GLFW_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIR})

cuda_add_executable(simple ${SOURCE_FILES})
#add_executable(simple ${SOURCE_FILES})
target_link_libraries(simple ${GLEW_LIBRARIES} ${GLFW_LIBRARY} ${OPENGL_LIBRARY} ${EXTRA_LIBRARIES} ${CUDA_LIBRARIES})
