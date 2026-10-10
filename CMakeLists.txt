cmake_minimum_required(VERSION 3.20)
project(npa_asr_app LANGUAGES C CXX)

set(ASR_VERSION "0.1.0" CACHE STRING "application version")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------- options
option(ASR_GUI     "build the ImGui desktop GUI"                ON)
option(ASR_TOOLS   "build test/benchmark tools"                 ON)
option(ASR_NATIVE  "single-arch binary tuned for the build machine, no runtime CPU dispatch" OFF)
# oneMKL DFTI for the 320-point mel FFT.  OFF by default on purpose: linking
# oneMKL makes the executable depend on the oneAPI runtime (on Windows
# mkl_rt.2.dll must be on PATH, i.e. you have to run setvars.bat before the app),
# while the built-in Bluestein FFT is portable and costs ~20 MFLOP/s of audio.
# scripts/build_linux.sh turns it ON because there the linker records an rpath.
option(ASR_MKL_FFT "use Intel oneMKL for the mel-spectrogram FFT when available" OFF)
option(ASR_FETCH_DEPS "download missing third-party sources at configure time" ON)

if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

include(cmake/oneapi.cmake)
include(cmake/deps.cmake)

# ---------------------------------------------------------------- ggml
# Two modes:
#  * default   : one binary carrying every x86 CPU backend variant and picking
#                the best at runtime (alderlake/AVX-VNNI on Raptor Lake, falling
#                back to haswell/AVX2, sandybridge/AVX, sse42, x64);
#  * ASR_NATIVE: tuned for the build machine only (fastest, no fallback).
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         ON  CACHE BOOL "" FORCE)
set(GGML_CCACHE         OFF CACHE BOOL "" FORCE)
set(GGML_LTO            OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS   OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS_3RD_PARTY OFF CACHE BOOL "" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)

if(ASR_NATIVE)
  set(GGML_NATIVE           ON  CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL       OFF CACHE BOOL "" FORCE)
  set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
  message(STATUS "ggml: native build (no runtime CPU dispatch)")
else()
  set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL       ON  CACHE BOOL "" FORCE)
  set(GGML_CPU_ALL_VARIANTS ON  CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS     ON  CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------- asr core
file(GLOB_RECURSE ASR_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/net/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/app/*.cpp)

add_library(asr_core STATIC ${ASR_SOURCES})
target_include_directories(asr_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(asr_core PUBLIC ggml)
target_compile_definitions(asr_core PUBLIC
  ASR_VERSION="${ASR_VERSION}"
  ASR_HAVE_GUI=$<BOOL:${ASR_GUI}>)
if(asr_have_mkl_fft)
  target_compile_definitions(asr_core PUBLIC ASR_HAVE_MKL_FFT=1)
  target_include_directories(asr_core PUBLIC ${MKL_INCLUDE_DIR})
  target_link_libraries(asr_core PUBLIC ${MKL_RT})
endif()

find_package(Threads REQUIRED)
target_link_libraries(asr_core PUBLIC Threads::Threads)

# libcurl (optional) - used by the Linux downloader when the headers exist.
find_package(CURL QUIET)
if(CURL_FOUND)
  message(STATUS "libcurl found: ${CURL_VERSION_STRING}")
  target_compile_definitions(asr_core PUBLIC ASR_HAVE_LIBCURL=1)
  target_link_libraries(asr_core PUBLIC CURL::libcurl)
endif()

# X11 / XTest / XRecord (Linux only)
if(UNIX AND NOT APPLE)
  find_package(X11 QUIET)
  if(X11_FOUND)
    target_compile_definitions(asr_core PUBLIC ASR_HAVE_X11=1)
    target_link_libraries(asr_core PUBLIC X11::X11)
    if(X11_XTest_FOUND)
      target_compile_definitions(asr_core PUBLIC ASR_HAVE_XTEST=1)
      target_link_libraries(asr_core PUBLIC ${X11_XTest_LIB})
      message(STATUS "XTest: ${X11_XTest_LIB}")
    endif()
    # FindX11 does not know about XRecord (it lives in libXtst)
    find_path(X11_XRECORD_INCLUDE_DIR X11/extensions/record.h)
    if(X11_XRECORD_INCLUDE_DIR AND X11_XTest_FOUND)
      target_compile_definitions(asr_core PUBLIC ASR_HAVE_XRECORD=1)
      message(STATUS "XRecord: ${X11_XRECORD_INCLUDE_DIR} (libXtst)")
    endif()
  endif()
endif()

# ---------------------------------------------------------------- GUI
if(ASR_GUI)
  set(IMGUI_DIR ${ASR_IMGUI_DIR})
  add_library(asr_imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(asr_imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
  target_link_libraries(asr_imgui PUBLIC glfw)
  if(WIN32)
    target_link_libraries(asr_imgui PUBLIC opengl32)
  endif()
  target_link_libraries(asr_core PUBLIC asr_imgui)
  if(UNIX AND NOT APPLE)
    find_package(OpenGL REQUIRED)
    target_link_libraries(asr_imgui PUBLIC OpenGL::GL)
  endif()
endif()

# ---------------------------------------------------------------- executable
add_executable(npa_asr src/main.cpp)
target_link_libraries(npa_asr PRIVATE asr_core)
if(APPLE)
  target_link_libraries(npa_asr PRIVATE
    "-framework CoreFoundation" "-framework CoreAudio" "-framework AudioToolbox"
    "-framework Cocoa" "-framework IOKit")
endif()
if(WIN32)
  target_link_libraries(npa_asr PRIVATE winhttp shlwapi ws2_32)
endif()

# keep every runtime artifact (ggml backend DLLs/SOs) next to the executable
set_target_properties(npa_asr PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
foreach(tgt ggml-base ggml-cpu x64 sse42 sandybridge ivybridge piledriver haswell
            skylakex cannonlake cascadelake icelake cooperlake zen4 alderlake sapphirerapids)
  if(TARGET ggml-cpu-${tgt})
    set_target_properties(ggml-cpu-${tgt} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  elseif(TARGET ${tgt})
    set_target_properties(${tgt} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  endif()
endforeach()
if(TARGET ggml-base)
  set_target_properties(ggml-base PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

if(ASR_TOOLS)
  add_subdirectory(tools)
endif()
