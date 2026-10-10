# Intel oneAPI (icx/icpx + oneMKL + OpenMP) detection.
#
# The compiler itself is selected by the build scripts, which source
#   Linux  : ~/dists/intel/oneapi/setvars.sh
#   Windows: C:\Program Files (x86)\Intel\oneAPI\setvars.bat
# before invoking CMake, so that icx/icpx/iomp5/mkl are on PATH.
#
# This module only *detects* what is available and reports it.

set(asr_have_mkl_fft FALSE)
set(asr_oneapi_root "")

if(DEFINED ENV{ONEAPI_ROOT})
  set(asr_oneapi_root "$ENV{ONEAPI_ROOT}")
elseif(DEFINED ENV{SETVARS_ROOT})
  set(asr_oneapi_root "$ENV{SETVARS_ROOT}")
elseif(EXISTS "$ENV{HOME}/dists/intel/oneapi/setvars.sh")
  set(asr_oneapi_root "$ENV{HOME}/dists/intel/oneapi")
elseif(EXISTS "C:/Program Files (x86)/Intel/oneAPI/setvars.bat")
  set(asr_oneapi_root "C:/Program Files (x86)/Intel/oneAPI")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  message(STATUS "oneAPI: IntelLLVM compiler ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
else()
  message(STATUS "oneAPI: compiler is ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
                 "(run the build script to use icx/icpx)")
endif()

if(asr_oneapi_root)
  message(STATUS "oneAPI: root ${asr_oneapi_root}")
endif()

# ---- oneMKL FFT (320-point real FFT for the log-mel front-end) -------------
if(ASR_MKL_FFT)
  set(MKL_SEARCH_HINTS
    "$ENV{MKLROOT}"
    "$ENV{MKLROOT}/lib"
    "$ENV{MKLROOT}/lib/intel64"
    "${asr_oneapi_root}/mkl/latest/lib/intel64"
    "${asr_oneapi_root}/mkl/latest/lib"
    "${asr_oneapi_root}/2025.3/mkl/lib/intel64"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64")

  find_library(MKL_RT NAMES mkl_rt mkl_rt.2 HINTS ${MKL_SEARCH_HINTS})
  find_path(MKL_INCLUDE_DIR NAMES mkl_dfti.h
    HINTS "$ENV{MKLROOT}/include" "${asr_oneapi_root}/mkl/latest/include"
          "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include")

  if(MKL_RT AND MKL_INCLUDE_DIR)
    set(asr_have_mkl_fft TRUE)
    message(STATUS "oneAPI: oneMKL FFT enabled (${MKL_RT})")
  else()
    message(STATUS "oneAPI: oneMKL not found - falling back to the built-in FFT "
                   "(MKL_RT=${MKL_RT} MKL_INCLUDE_DIR=${MKL_INCLUDE_DIR})")
  endif()
endif()
