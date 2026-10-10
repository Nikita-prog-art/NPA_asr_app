cmake_minimum_required(VERSION 3.20)
project(npa_asr_app LANGUAGES C CXX)

set(ASR_VERSION "0.1.0" CACHE STRING "application version")

# ---------------------------------------------------------------- build id
# The source revision embedded in the binary (src/app/build_info.cpp).  The
# application uses it to tell builds apart: it offers to update an installed
# copy when the revision differs, and stays quiet when the same revision is
# just rebuilt.  Pass -DASR_BUILD_ID=... when the sources were exported without
# .git (scripts/build_windows_from_wsl.sh does that, since MSVC cannot compile
# from a UNC path and the sources travel to Windows as a tarball).
set(ASR_BUILD_ID "" CACHE STRING "source revision embedded in the binary")
if(NOT ASR_BUILD_ID)
  execute_process(COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short HEAD
                  OUTPUT_VARIABLE ASR_GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE ASR_GIT_RC)
  if(ASR_GIT_RC EQUAL 0 AND ASR_GIT_REV)
    set(ASR_BUILD_ID "${ASR_GIT_REV}")
    # Local modifications are a different build as well: mark them.  Untracked
    # files are ignored, so build directories and helper scripts do not count.
    execute_process(COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" status --porcelain --untracked-files=no
                    OUTPUT_VARIABLE ASR_GIT_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET RESULT_VARIABLE ASR_GIT_RC)
    if(ASR_GIT_RC EQUAL 0 AND ASR_GIT_DIRTY)
      set(ASR_BUILD_ID "${ASR_BUILD_ID}-dirty")
    endif()
  else()
    set(ASR_BUILD_ID "${ASR_VERSION}")
    message(STATUS "build id: git not available, using the version ${ASR_BUILD_ID}")
  endif()
endif()
message(STATUS "build id: ${ASR_BUILD_ID}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Everything we build ourselves (ggml, imgui, glfw, the application) is linked
# statically, so the result is one self-contained executable and "installing"
# the application means copying a single file.  On Windows this also selects the
# static C runtime (/MT), so the target machine needs no Visual C++
# redistributable; on Linux the C++ runtime is linked in as well, which leaves
# only the system libraries (libX11, libGL, libcurl) dynamic.
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ---------------------------------------------------------------- options
option(ASR_GUI     "build the ImGui desktop GUI"                ON)
option(ASR_TOOLS   "build test/benchmark tools"                 ON)
option(ASR_FETCH_DEPS "download missing third-party sources at configure time" ON)
# libcurl pulls in a large part of a Linux userland (openssl, gnutls, krb5, ldap,
# ...).  With -DASR_LIBCURL=OFF the downloader spawns the `curl`/`wget` command
# line tool instead and the executable depends on X11/OpenGL/libc only.
option(ASR_LIBCURL "use libcurl for model downloads on Linux" ON)
# ggml's own thread pool is used by default; OpenMP adds a runtime dependency
# (libgomp.so / vcomp140.dll) and measured no faster, so it stays off.
option(ASR_OPENMP  "let ggml use OpenMP (adds a libgomp/vcomp runtime dependency)" OFF)
set(ASR_CPU_ARCH "x86-64-v3" CACHE STRING
    "x86-64 baseline the binary is compiled for: native | x86-64-v3 | x86-64-v2 | x86-64")
# Kept as a shorthand for -DASR_CPU_ARCH=native.
option(ASR_NATIVE  "build for the machine that compiles the project" OFF)

if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
  add_link_options(-static-libgcc -static-libstdc++)
endif()

# ---------------------------------------------------------------- CPU baseline
# A statically linked binary cannot pick a CPU backend at run time, so the
# instruction set is fixed when the project is configured.  The default,
# x86-64-v3 (AVX2 + FMA + F16C + BMI2), runs on every Intel CPU since Haswell
# (2013) and every AMD Zen; "native" is fastest but only runs on the machine
# that compiled it, and "x86-64-v2"/"x86-64" go further back.
if(ASR_NATIVE)
  set(ASR_CPU_ARCH "native" CACHE STRING "" FORCE)
endif()

set(ASR_ARCH_FLAG "")
if(MSVC)
  if(ASR_CPU_ARCH STREQUAL "native" OR ASR_CPU_ARCH STREQUAL "x86-64-v3")
    set(ASR_ARCH_FLAG "/arch:AVX2")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  if(ASR_CPU_ARCH STREQUAL "native")
    set(ASR_ARCH_FLAG "-march=native")
  elseif(ASR_CPU_ARCH MATCHES "^x86-64")
    set(ASR_ARCH_FLAG "-march=${ASR_CPU_ARCH}")
  else()
    message(FATAL_ERROR "ASR_CPU_ARCH must be native, x86-64-v3, x86-64-v2 or x86-64")
  endif()
endif()
if(ASR_ARCH_FLAG)
  add_compile_options(${ASR_ARCH_FLAG})
endif()
message(STATUS "CPU baseline: ${ASR_CPU_ARCH} ${ASR_ARCH_FLAG}")

# ggml enables AVX2 unconditionally on x86 when it is not told otherwise, so its
# instruction set switches are kept in sync with ASR_CPU_ARCH.
if(ASR_CPU_ARCH STREQUAL "native")
  set(ASR_GGML_NATIVE ON)
  set(ASR_ISA_SSE42 ON)
  set(ASR_ISA_AVX   ON)
  set(ASR_ISA_AVX2  ON)
  set(ASR_ISA_FMA   ON)
  set(ASR_ISA_F16C  ON)
  set(ASR_ISA_BMI2  ON)
else()
  set(ASR_GGML_NATIVE OFF)
  set(ASR_ISA_SSE42 OFF)
  set(ASR_ISA_AVX   OFF)
  set(ASR_ISA_AVX2  OFF)
  set(ASR_ISA_FMA   OFF)
  set(ASR_ISA_F16C  OFF)
  set(ASR_ISA_BMI2  OFF)
  if(ASR_CPU_ARCH STREQUAL "x86-64-v3")
    set(ASR_ISA_SSE42 ON)
    set(ASR_ISA_AVX   ON)
    set(ASR_ISA_AVX2  ON)
    set(ASR_ISA_FMA   ON)
    set(ASR_ISA_F16C  ON)
    set(ASR_ISA_BMI2  ON)
  elseif(ASR_CPU_ARCH STREQUAL "x86-64-v2")
    set(ASR_ISA_SSE42 ON)
    set(ASR_ISA_AVX   ON)
  elseif(NOT ASR_CPU_ARCH STREQUAL "x86-64")
    message(FATAL_ERROR "ASR_CPU_ARCH must be native, x86-64-v3, x86-64-v2 or x86-64")
  endif()
endif()

include(cmake/deps.cmake)

# ---------------------------------------------------------------- ggml
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE         OFF CACHE BOOL "" FORCE)
set(GGML_LTO            OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS   OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS_3RD_PARTY OFF CACHE BOOL "" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         ${ASR_OPENMP} CACHE BOOL "" FORCE)
# Static single-ISA CPU backend: no runtime-dispatched modules next to the
# executable, no BUILD_SHARED_LIBS, no dlopen of ggml-cpu-*.so/.dll.
set(GGML_BACKEND_DL       OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(GGML_STATIC           OFF CACHE BOOL "" FORCE)  # would add a global -static
set(GGML_NATIVE           ${ASR_GGML_NATIVE} CACHE BOOL "" FORCE)
set(GGML_SSE42            ${ASR_ISA_SSE42} CACHE BOOL "" FORCE)
set(GGML_AVX              ${ASR_ISA_AVX}   CACHE BOOL "" FORCE)
set(GGML_AVX2             ${ASR_ISA_AVX2}  CACHE BOOL "" FORCE)
set(GGML_FMA              ${ASR_ISA_FMA}   CACHE BOOL "" FORCE)
set(GGML_F16C             ${ASR_ISA_F16C}  CACHE BOOL "" FORCE)
set(GGML_BMI2             ${ASR_ISA_BMI2}  CACHE BOOL "" FORCE)
set(GGML_AVX_VNNI         OFF CACHE BOOL "" FORCE)
set(GGML_AVX512           OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ggml is third-party code compiled with our global -Wall/-Wextra; silence it so
# that real warnings in src/ remain visible in the build output.
foreach(ggml_target ggml ggml-base ggml-cpu)
  if(TARGET ${ggml_target})
    if(MSVC)
      target_compile_options(${ggml_target} PRIVATE /w)
    else()
      target_compile_options(${ggml_target} PRIVATE -w)
    endif()
  endif()
endforeach()

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

# The revision is a generated header rather than a target-wide definition so
# that a new commit recompiles src/app/build_info.cpp only.  configure_file()
# leaves the file (and therefore the build) alone while the value is unchanged.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_id.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/npa_asr_build_id.h @ONLY)
target_include_directories(asr_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(asr_core PUBLIC ggml)
target_compile_definitions(asr_core PUBLIC
  ASR_VERSION="${ASR_VERSION}"
  ASR_CPU_BASELINE="${ASR_CPU_ARCH}"
  ASR_HAVE_GUI=$<BOOL:${ASR_GUI}>)

find_package(Threads REQUIRED)
target_link_libraries(asr_core PUBLIC Threads::Threads)

# libcurl (optional) - used by the Linux downloader when the headers exist.
if(ASR_LIBCURL)
  find_package(CURL QUIET)
  if(CURL_FOUND)
    message(STATUS "libcurl found: ${CURL_VERSION_STRING}")
    target_compile_definitions(asr_core PUBLIC ASR_HAVE_LIBCURL=1)
    target_link_libraries(asr_core PUBLIC CURL::libcurl)
  else()
    message(STATUS "libcurl not found: downloads will spawn curl/wget")
  endif()
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
  # shell32: Shell_NotifyIcon (the notification-area icon in platform_win.cpp)
  target_link_libraries(npa_asr PRIVATE winhttp shlwapi shell32 ws2_32)
  # Application manifest: Per-Monitor-V2 DPI awareness (a statically linked GLFW
  # cannot carry its own manifest), long path support, asInvoker.
  if(MSVC)
    # The linker already emits a default manifest (/MANIFEST:EMBED,ID=1), so ours
    # is merged into it with /MANIFESTINPUT.  Adding it as a resource instead
    # fails with CVT1100 "duplicate resource, type MANIFEST, name 1".
    target_link_options(npa_asr PRIVATE
      "/MANIFEST:EMBED"
      "/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/cmake/npa_asr.manifest")
  else()
    # MinGW and friends: windres embeds it as resource id 1.
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/npa_asr.rc.in
                   ${CMAKE_CURRENT_BINARY_DIR}/npa_asr.rc @ONLY)
    target_sources(npa_asr PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/npa_asr.rc)
  endif()
endif()

set_target_properties(npa_asr PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

if(ASR_TOOLS)
  add_subdirectory(tools)
endif()
