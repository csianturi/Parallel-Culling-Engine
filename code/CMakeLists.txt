cmake_minimum_required(VERSION 3.18)

set(CMAKE_TOOLCHAIN_FILE "C:/CS242Project/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(FrustumCulling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(culling_benchmark 
    src/main.cpp 
    src/culling.cu 
)

target_link_libraries(culling_benchmark PRIVATE 
    glfw 
    glad::glad 
    glm::glm 
    OpenGL::GL
)