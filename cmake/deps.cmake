# Third-party dependencies.
#
# Everything is pinned.  Sources already present under third_party/ are used
# as-is, otherwise they are downloaded at configure time (ASR_FETCH_DEPS=ON).
# Run scripts/fetch_deps.sh to pre-populate the directory for offline builds.

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/third_party/_fetched)

set(ASR_GGML_COMMIT ffa4e8b80930029a35991f94e7c8a93cd67730ab)  # ggml 0.26.0
set(ASR_IMGUI_TAG  v1.92.9b)
set(ASR_GLFW_TAG   3.4)
set(ASR_MINI_TAG   0.11.25)
set(ASR_JSON_TAG   v3.11.3)

function(asr_need_dir var path url)
  if(EXISTS "${path}/CMakeLists.txt" OR EXISTS "${path}/CMakeLists.txt.in")
    set(${var} "${path}" PARENT_SCOPE)
    return()
  endif()
  if(NOT ASR_FETCH_DEPS)
    message(FATAL_ERROR "missing ${path}; run scripts/fetch_deps.sh or configure with -DASR_FETCH_DEPS=ON")
  endif()
  message(STATUS "fetching ${url} -> ${path}")
  FetchContent_Declare(${var}_src URL ${url} SOURCE_DIR ${path})
  FetchContent_Populate(${var}_src)
  set(${var} "${path}" PARENT_SCOPE)
endfunction()

function(asr_need_file path url)
  if(EXISTS "${path}")
    return()
  endif()
  if(NOT ASR_FETCH_DEPS)
    message(FATAL_ERROR "missing ${path}; run scripts/fetch_deps.sh or configure with -DASR_FETCH_DEPS=ON")
  endif()
  message(STATUS "fetching ${url} -> ${path}")
  file(DOWNLOAD ${url} ${path} STATUS st TLS_VERIFY ON)
  list(GET st 0 code)
  if(NOT code EQUAL 0)
    list(GET st 1 msg)
    file(REMOVE ${path})
    message(FATAL_ERROR "download failed: ${msg}")
  endif()
endfunction()

# ---- ggml (the inference core) ---------------------------------------------
asr_need_dir(ASR_GGML_DIR ${CMAKE_SOURCE_DIR}/third_party/ggml
  https://github.com/ggml-org/ggml/archive/${ASR_GGML_COMMIT}.tar.gz)

# ---- single-header libraries -----------------------------------------------
asr_need_file(${CMAKE_SOURCE_DIR}/third_party/miniaudio.h
  https://raw.githubusercontent.com/mackron/miniaudio/${ASR_MINI_TAG}/miniaudio.h)
asr_need_file(${CMAKE_SOURCE_DIR}/third_party/json.hpp
  https://github.com/nlohmann/json/releases/download/${ASR_JSON_TAG}/json.hpp)

# ---- GUI -------------------------------------------------------------------
if(ASR_GUI)
  asr_need_dir(ASR_IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui
    https://github.com/ocornut/imgui/archive/refs/tags/${ASR_IMGUI_TAG}.tar.gz)
  asr_need_dir(ASR_GLFW_DIR ${CMAKE_SOURCE_DIR}/third_party/glfw
    https://github.com/glfw/glfw/archive/refs/tags/${ASR_GLFW_TAG}.tar.gz)

  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_WAYLAND  OFF CACHE BOOL "" FORCE)
  # ggml needs BUILD_SHARED_LIBS=ON for its runtime-dispatched CPU backends, and
  # that would otherwise turn GLFW into glfw3.dll/libglfw.so - force static so
  # the application has no extra runtime dependency.
  set(GLFW_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
  add_subdirectory(${ASR_GLFW_DIR} ${CMAKE_BINARY_DIR}/glfw EXCLUDE_FROM_ALL)

  # belt and braces: if a shared GLFW is ever produced, keep it next to the exe
  if(TARGET glfw)
    set_target_properties(glfw PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  endif()
endif()
