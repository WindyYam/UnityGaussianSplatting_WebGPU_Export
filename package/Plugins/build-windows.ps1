# PowerShell build script for Windows NativeSorting DLL
# Requires MinGW-w64 to be installed and in PATH

Write-Host "Building NativeSorting.dll for Windows..." -ForegroundColor Green

# Set source and output paths
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$sourceDir = Join-Path $scriptPath "WebGL"
$outputDir = Join-Path $scriptPath "x86_64"
$sourceFile = Join-Path $sourceDir "NativeSorting.c"
$outputFile = Join-Path $outputDir "NativeSorting.dll"

# Create output directory if it doesn't exist
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Check if source file exists
if (-not (Test-Path $sourceFile)) {
    Write-Error "Source file not found: $sourceFile"
    exit 1
}

# Check if gcc is available
try {
    $gccVersion = & gcc --version 2>&1
    Write-Host "Found GCC: $($gccVersion[0])" -ForegroundColor Cyan
} catch {
    Write-Error "gcc not found in PATH. Please install MinGW-w64 and add it to your PATH."
    Write-Host "Download from: https://www.mingw-w64.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Compile the DLL
Write-Host "Compiling with gcc..." -ForegroundColor Yellow
try {
    $compileArgs = @(
        "-shared",
        "-fPIC", 
        "-o", $outputFile,
        $sourceFile,
        "-lpthread",
        "-O2"
    )
    
    & gcc $compileArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "SUCCESS: NativeSorting.dll built successfully!" -ForegroundColor Green
        Write-Host "Location: $outputFile" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Restart Unity to reload the plugin"
        Write-Host "2. Check the Console for 'NativeSorting: Initialized with X worker threads' message"
        Write-Host "3. Configure the DLL import settings in Unity Inspector if needed"
    } else {
        throw "Compilation failed"
    }
} catch {
    Write-Host ""
    Write-Warning "Failed to compile with pthread. Trying fallback build without pthread..."
    
    try {
        $fallbackArgs = @(
            "-shared",
            "-fPIC", 
            "-o", $outputFile,
            $sourceFile,
            "-DSINGLE_THREADED",
            "-O2"
        )
        
        & gcc $fallbackArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: NativeSorting.dll built successfully (single-threaded mode)" -ForegroundColor Green
            Write-Host "Note: This build uses single-threaded fallback. For better performance, install pthread support." -ForegroundColor Yellow
        } else {
            throw "Fallback compilation also failed"
        }
    } catch {
        Write-Error "Both compilation attempts failed. Check that gcc is properly installed and NativeSorting.c exists."
        exit 1
    }
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
