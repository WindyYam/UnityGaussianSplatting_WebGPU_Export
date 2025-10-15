@echo off
REM Build script for Windows NativeSorting DLL
REM Requires MinGW-w64 to be installed and in PATH

echo Building NativeSorting.dll for Windows...

REM Set source and output paths
set SOURCE_DIR=%~dp0WebGL
set OUTPUT_DIR=%~dp0x86_64

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Check if gcc is available
gcc --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: gcc not found in PATH. Please install MinGW-w64 and add it to your PATH.
    echo Download from: https://www.mingw-w64.org/downloads/
    pause
    exit /b 1
)

REM Compile the DLL
echo Compiling with gcc...
gcc -shared -fPIC -o "%OUTPUT_DIR%\NativeSorting.dll" "%SOURCE_DIR%\NativeSorting.c" -lpthread -O2

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: NativeSorting.dll built successfully!
    echo Location: %OUTPUT_DIR%\NativeSorting.dll
    echo.
    echo Next steps:
    echo 1. Restart Unity to reload the plugin
    echo 2. Check the Console for "NativeSorting: Initialized with X worker threads" message
    echo 3. Configure the DLL import settings in Unity Inspector if needed
) else (
    echo.
    echo ERROR: Failed to compile NativeSorting.dll
    echo.
    echo Trying fallback build without pthread...
    gcc -shared -fPIC -o "%OUTPUT_DIR%\NativeSorting.dll" "%SOURCE_DIR%\NativeSorting.c" -DSINGLE_THREADED -O2
    
    if %ERRORLEVEL% EQU 0 (
        echo SUCCESS: NativeSorting.dll built successfully (single-threaded mode)
        echo Note: This build uses single-threaded fallback. For better performance, install pthread support.
    ) else (
        echo ERROR: Both compilation attempts failed.
        echo Check that NativeSorting.c exists and gcc is properly installed.
    )
)

echo.
pause
