cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
project(rt VERSION 1.0.0)

add_library(
    geo
    geo.hh
    geo.cc
    geo.h
    geo.c
)

set(SDL_TEST OFF)
add_subdirectory(SDL)
include_directories(${CMAKE_SOURCE_DIR}/SDL/include)

add_executable(rt rt.cc)
target_link_libraries(rt PRIVATE geo PRIVATE SDL2::SDL2)
