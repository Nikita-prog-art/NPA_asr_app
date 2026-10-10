@echo off
rem Build npa_asr on Windows.
rem
rem   scripts\build_windows.cmd [extra cmake args...]
rem
rem Environment:
rem   ONEAPI_ROOT   oneAPI installation (default: C:\Program Files (x86)\Intel\oneAPI)
rem   ASR_COMPILER  cl | icx   (default: cl, the best tested path for ggml on MSVC)
rem   ASR_MKL_FFT   ON to link oneMKL for the mel FFT (default: OFF, see README)
rem   ASR_NATIVE    ON to build a single-arch binary for this machine only
rem   BUILD_DIR     build directory (default: <repo>\build-windows)
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
if "%BUILD_DIR%"=="" set "BUILD_DIR=%ROOT%\build-windows"
if "%ONEAPI_ROOT%"=="" set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
if "%ASR_COMPILER%"=="" set "ASR_COMPILER=cl"

rem ---- Visual Studio environment (MSVC + ninja + cmake come with VS) --------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
)
if defined VSPATH (
    echo [build] Visual Studio: !VSPATH!
    call "!VSPATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
) else (
    echo [build] vswhere/VS not found - assuming cl.exe is already on PATH
)

rem ---- oneAPI (optional: only for the Intel compiler or oneMKL) ------------
rem oneMKL is opt-in because it makes the exe depend on mkl_rt.2.dll at runtime
rem (you would have to run setvars.bat before starting the app), and it is worth
rem about 0.2% of the total runtime - see README.
if /I "%ASR_COMPILER%"=="icx" goto :oneapi
if /I "%ASR_MKL_FFT%"=="ON" goto :oneapi
goto :after_oneapi
:oneapi
if exist "%ONEAPI_ROOT%\setvars.bat" (
    echo [build] oneAPI: %ONEAPI_ROOT%
    call "%ONEAPI_ROOT%\setvars.bat" intel64 >nul 2>&1
) else (
    echo [build] oneAPI not found at %ONEAPI_ROOT%
)
:after_oneapi

set "CC_BIN=cl"
set "CXX_BIN=cl"
if /I "%ASR_COMPILER%"=="icx" (
    set "CC_BIN=icx"
    set "CXX_BIN=icx"
    where icx >nul 2>&1 || (
        echo [build] icx not found, falling back to cl
        set "CC_BIN=cl"
        set "CXX_BIN=cl"
    )
)

rem oneMKL needs mkl_rt.2.dll on PATH at runtime, so it is opt-in on Windows:
rem   set ASR_MKL_FFT=ON   (then run setvars.bat before npa_asr.exe)
set "MKL_ARG=-DASR_MKL_FFT=OFF"
if /I "%ASR_MKL_FFT%"=="ON" set "MKL_ARG=-DASR_MKL_FFT=ON"

set "NATIVE_ARG="
if /I not "%ASR_NATIVE%"=="OFF" set "NATIVE_ARG=-DASR_NATIVE=ON"

rem ggml compiles its CPU backend once per ISA variant and the big translation
rem units (ggml-cpu/ops.cpp with AVX512) need ~2 GB of RAM each: budget ~2 GB
rem per job so a memory-limited VM/WSL does not get swapped to death.
set "JOBS=%ASR_JOBS%"
if "%JOBS%"=="" (
    for /f %%m in ('powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)" 2^>nul') do set "RAM_GB=%%m"
    if not defined RAM_GB set "RAM_GB=8"
    set /a JOBS=%RAM_GB%/2
    if %JOBS% LSS 1 set "JOBS=1"
)
echo [build] parallel jobs: %JOBS%

echo [build] compiler: %CC_BIN%
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=%CC_BIN% ^
    -DCMAKE_CXX_COMPILER=%CXX_BIN% ^
    %NATIVE_ARG% %MKL_ARG% %*
if errorlevel 1 goto :fail

cmake --build "%BUILD_DIR%" -j %JOBS%
if errorlevel 1 goto :fail

echo.
echo [build] done: %BUILD_DIR%\bin\npa_asr.exe
echo [build] note: keep the ggml-cpu-*.dll files next to the executable
exit /b 0

:fail
echo [build] FAILED
exit /b 1
