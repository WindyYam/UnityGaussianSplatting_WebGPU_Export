# Unity Gaussian Splatting (WebGPU-Oriented Fork)

**Live WebGPU Demo:** https://windyyam.github.io/unity_splat_test_deploy/

This repository is a fork of `aras-p/UnityGaussianSplatting`. Many thanks to @aras-p for the original project and the work that made this fork possible: https://github.com/aras-p/UnityGaussianSplatting

This fork originally followed the `stochastic` branch of the original project and experimented with stochastic sort-free alpha blend. It was heavily modified to build on WebGPU — including removal of the compute shader path — but the stochastic approach introduced noise that degrades the visual quality. As a result, the implementation reverted to CPU-side per-node sorting, augmented with node culling, caching of per-node orderings, and distance-based ordering to mitigate cost and visual artifacts.

As a result, this is a portability-focused fork primarily targeting WebGPU with high visual fidelity splat rendering that works well on constrained / upcoming WebGPU platforms. To maximize compatibility on Web and WASM, the implementation favors CPU-side hierarchical culling and small per-node sorts rather than relying on large, complex GPU-only global sorts.

**Native Threading Support**: The implementation includes native threading for parallel sorting with platform-specific optimizations. On WebGL, this uses Emscripten's pthread support for true parallel processing in WebAssembly where Unity's Task system is not available. On desktop platforms (Windows/macOS/Linux), the system uses Unity's Task system by default, which provides excellent cross-platform performance without additional dependencies. Native threading libraries can be optionally compiled for desktop platforms to squeeze out additional performance, but this requires manual compilation and is not necessary for most use cases.

## Highlights
- **Smart Threading Strategy**: WebGL uses native threading (Emscripten pthreads) where Unity Tasks aren't available. Desktop platforms use Unity's Task system by default for optimal cross-platform compatibility, with optional native libraries for advanced users.
- **Hierarchical octree build** (configurable depth & leaf capacity) + outlier bucket.
- **Frustum culling** at node granularity drastically reduces sort domain.
- **Non-blocking parallel sorts**: Per-node sorts run asynchronously in the background, updating cached orderings when complete so rendering is never blocked by sorting operations.
- **Distance-based ordering**: Node centers are pre-ordered by distance to the camera position, and individual splats inside each node are also sorted by distance (not by camera forward). This preserves the sort order when the camera rotates in place, no re-sort needed.
- **WebGPU-first design**: CPU-friendly algorithms with intelligent threading ensure high performance and portability across all platforms, including constrained WebGPU environments.

## Quick Start
1. Import / build a Gaussian Splat asset (PLY / SPZ) via the provided editor tooling.
2. Octree builds once (respect `maxDepth`, `maxSplatsPerLeaf`). Outliers separated automatically during preprocessing: very far-away sparse splats (e.g. sky pixels, distant noise, or background samples produced by MCMC strategies) are detected and placed into an outlier bucket and treated as background. This lets the octree build focus on the main dense splat region — improving node efficiency and reducing per-node sort cost. The aggressiveness of the outlier filter can be tuned via the "scene splat ratio" setting to control how many splats are included in the main octree build versus moved to the outlier bucket.

   Tip: You can visualize the octree node bounds and leaf centers in the Unity Scene view by enabling the "Draw Gizmos" checkbox in the Gaussian Splat settings script. This overlay is useful for diagnosing outliers and tuning `maxDepth`, `maxSplatsPerLeaf`, and the scene splat ratio.
   3. Each frame (alpha blend mode):
   - **Frustum cull nodes** to reduce processing overhead.
   - **Gather visible leaves** based on camera frustum.
   - **Sort visible leaves** by node center distance for optimal rendering order.
   - **Kick off parallel sorts** in the background (non-blocking). Uses native threading on WebGL (Emscripten pthreads), Unity Task system on desktop by default.
   - **Asynchronous result collection**: Per-leaf sorting runs in worker threads and updates per-node cached orderings when complete, ensuring main thread is never blocked.
   - **Concatenate and upload**: Currently available per-node indices are assembled by leaf sort order into a single GPU buffer consumed by the renderer; updated orderings from completed background sorts are seamlessly picked up on subsequent frames.

Note: The GaussianExample-URP package includes a ready-to-play scene named "Barangaroo".

## Threading Architecture
This implementation features a smart cross-platform threading strategy optimized for each platform's capabilities:

**WebGL/WebAssembly**: Uses native threading (Emscripten's pthread implementation) compiled to WebAssembly with SharedArrayBuffer support. This is essential since Unity's Task system is not available on WebGL, enabling genuine multi-threaded execution in browsers.

⚠️ **WebGL Threading Requirements**:
1. **Unity Project Settings**: Enable "Native C++ Threads Support" in Project Settings > XR Plug-in Management > WebGL Settings
2. **HTTP Headers**: Serve your WebGL build with these headers to enable SharedArrayBuffer:
   ```
   Cross-Origin-Opener-Policy: same-origin
   Cross-Origin-Embedder-Policy: require-corp
   ```
3. **Browser Support**: Modern browsers with SharedArrayBuffer support (Chrome 68+, Firefox 79+, Safari 15.2+)

**Desktop Platforms**: Uses Unity's Task system by default, which provides excellent performance and cross-platform compatibility without external dependencies. This is not a performance bottleneck and offers the best balance of performance and maintainability.

**Optional Native Libraries**: Advanced users can compile platform-native libraries for desktop platforms to achieve maximum performance:
1. **Native threading** (maximum performance) - requires manual compilation of native libraries
2. **Unity Task system** (default, excellent performance) - works out of the box on all desktop platforms
3. **Sequential processing** (compatibility fallback) - used only when parallel processing fails

**Non-Blocking Design**: All sorting operations run asynchronously in background threads, ensuring the main rendering thread is never blocked. Results are collected and applied only when jobs complete, maintaining smooth frame rates even during intensive sorting operations.

## Runtime Settings
| Setting | Description | Guidance |
|---------|-------------|----------|
| `maxDepth`, `maxSplatsPerLeaf` | Octree granularity | Tune so typical visible leaf has ~1024 splats |
| `enableParallelSorting` | Toggle native threading | Enabled by default, uses native threads on all supported platforms |
| `parallelSortThreads` | Native worker thread count | Auto-detected from hardware, typically 4-8 threads optimal |

## Threading Implementation by Platform
| Platform | Default Threading | Requirements | Performance |
|----------|-------------------|--------------|-------------|
| WebGL | Native (Emscripten pthreads) | Unity native C++ threads + COOP/COEP headers | Essential - Unity Tasks not available |
| Windows | Unity Task System | None (optional: NativeSorting.dll) | Excellent by default |
| macOS | Unity Task System | None (optional: NativeSorting.dylib) | Excellent by default |
| Linux | Unity Task System | None (optional: libNativeSorting.so) | Excellent by default |

**WebGL Native Threading Setup**:
1. **Unity Settings**:
   - Open Project Settings > XR Plug-in Management > WebGL Settings
   - Enable "Native C++ Threads Support"
   - This enables SharedArrayBuffer and threading in your WebGL build

2. **Web Server Configuration**:
   ```apache
   # Apache .htaccess example
   Header always set Cross-Origin-Opener-Policy "same-origin"
   Header always set Cross-Origin-Embedder-Policy "require-corp"
   ```
   ```nginx
   # Nginx example
   add_header Cross-Origin-Opener-Policy "same-origin" always;
   add_header Cross-Origin-Embedder-Policy "require-corp" always;
   ```
   ```javascript
   // Node.js/Express example
   app.use((req, res, next) => {
     res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
     res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
     next();
   });
   ```

3. **Testing**: Check browser developer console for "SharedArrayBuffer is defined: true" to verify threading support is active.

**Building Native Libraries (Optional)**: Desktop users who want maximum performance can compile native libraries using the provided build scripts in `package/Plugins/`. See `docs/building-windows-dll.md` and `docs/native-threading-cross-platform.md` for detailed instructions. **Note**: This is not required for good performance - Unity's Task system is already highly optimized and cross-platform.

Recommended defaults and guidance:
- **Keep node sizes reasonable** so per-leaf sorts stay cheap. A good starting point for many scenes is `maxDepth = 8` and `maxSplatsPerLeaf = 1024`.
- **For the outlier filter** (scene splat ratio), try `0.90`–`0.95` to push very sparse, far-away splats into the outlier bucket; if your scene is very clean (no distant noise) you can set this to `1.0`.
- **Threading behavior** varies by platform: WebGL uses native threading when properly configured with Unity native C++ threads and COOP/COEP headers, desktop uses Unity Tasks by default (recommended), with optional native libraries for advanced users seeking maximum performance.

## Roadmap / Limitations
- **GPU acceleration**: For extreme splat counts (> tens of millions), a global GPU radix sort could provide additional performance benefits on high-end hardware.
- **Mobile platform support**: Native threading libraries for iOS and Android could be added to extend performance benefits to mobile platforms.
- **Visible-node ordering**: The current visible-node sort is an approximation for splat ordering and may benefit from more sophisticated ordering algorithms for complex scenes.

## License
See `LICENSE.md`.

Happy splatting.
