cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
project(Raytracing VERSION 1.0.0)

add_library(
    Geometry
    geo.hh
    geo.cc
    geo.h
    geo.c
)

add_executable(Raytracing rt.cc)
target_link_libraries(Raytracing PRIVATE Geometry)