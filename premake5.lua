-- premake5.lua
require("premake5-cuda/premake5-cuda")

workspace "RayTracing"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "RayTracing"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "RayTracing"

links { "$(VULKAN_SDK)/lib/vulkan-1.lib" }
includedirs { "$(VULKAN_SDK)/include/" }
