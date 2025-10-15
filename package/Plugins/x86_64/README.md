# Windows x64 Native Libraries

Place the compiled Windows native library here:
- `NativeSorting.dll` - The main native threading library

## Building the DLL

See `docs/building-windows-dll.md` for detailed build instructions.

Quick build command (requires MinGW):
```bash
gcc -shared -fPIC -o NativeSorting.dll ../WebGL/NativeSorting.c -lpthread -O2
```

## Plugin Settings

After placing the DLL here, configure Unity import settings:
- Platform: Standalone Windows (Win64)
- CPU: x86_64
- Editor platform: Windows x86_64

The native threading system will automatically detect and use this library when available.
