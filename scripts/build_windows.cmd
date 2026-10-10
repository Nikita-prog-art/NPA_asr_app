@echo off
rem Build npa_asr on Windows.
rem
rem   scripts\build_windows.cmd [extra cmake args...]
rem
rem Environment:
rem   ASR_CPU_ARCH  x86-64-v3 (default) | native | x86-64-v2 | x86-64
rem                 native = tuned for this machine only, no fallback
rem   ASR_NATIVE    ON is a shorthand for ASR_CPU_ARCH=native
rem   ASR_JOBS      parallel build jobs (default: derived from the RAM size)
rem   BUILD_DIR     build directory (default: <repo>\build-windows)
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
if "%BUILD_DIR%"=="" set "BUILD_DIR=%ROOT%\build-windows"

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

set "ARCH_ARG="
if not "%ASR_CPU_ARCH%"=="" set "ARCH_ARG=-DASR_CPU_ARCH=%ASR_CPU_ARCH%"
rem ASR_NATIVE is unset by default, so only treat it as ON when it is defined
rem and not "OFF" - otherwise an unset variable would enable native builds.
if defined ASR_NATIVE if /I not "%ASR_NATIVE%"=="OFF" set "ARCH_ARG=-DASR_NATIVE=ON"

rem One ggml CPU backend is compiled for a fixed ISA baseline, so a job needs
rem roughly 1 GB; budget 2 GB per job anyway so a memory-limited machine does
rem not get swapped to death.  ASR_JOBS overrides the heuristic.
rem NOTE: inside a parenthesised block %VAR% is expanded when the block is
rem parsed, so the values set inside it must be read with !VAR! (delayed
rem expansion is enabled above) - otherwise cmd.exe fails with
rem "1 was unexpected at this time".
set "JOBS=%ASR_JOBS%"
if not defined JOBS (
    for /f %%m in ('powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)" 2^>nul') do set "RAM_GB=%%m"
    if not defined RAM_GB set "RAM_GB=8"
    set /a JOBS=!RAM_GB!/2
    if !JOBS! LSS 1 set "JOBS=1"
)
echo [build] parallel jobs: %JOBS%

echo [build] compiler: cl
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    %ARCH_ARG% %*
if errorlevel 1 goto :fail

cmake --build "%BUILD_DIR%" -j %JOBS%
if errorlevel 1 goto :fail

rem ninja/cmake do not always surface a failed link through errorlevel, so the
rem result is verified independently before claiming success.
if not exist "%BUILD_DIR%\bin\npa_asr.exe" (
    echo [build] FAILED: %BUILD_DIR%\bin\npa_asr.exe was not produced
    exit /b 1
)

echo.
echo [build] done: %BUILD_DIR%\bin\npa_asr.exe
echo [build] the executable is self-contained: no DLLs are needed next to it
exit /b 0

:fail
echo [build] FAILED
exit /b 1
