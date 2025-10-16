# Unity Gaussian Splatting (WebGPU Fork)

**Live Demo:** https://windyyam.github.io/unity_splat_test_deploy/

A WebGPU-focused fork of [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) optimized for high-fidelity Gaussian splat rendering on web platforms and constrained environments.

**Key Features:**
- WebGPU-first design with hierarchical CPU-side culling and sorting
- Cross-platform threading: native threading on WebGL, Unity Tasks on desktop
- Non-blocking parallel sorting with octree optimization
- High visual fidelity without GPU compute shader dependencies

## Features
- **Cross-platform threading**: Native threading on WebGL, Unity Tasks on desktop
- **Hierarchical octree** with configurable depth and outlier filtering
- **Frustum culling** reduces processing overhead
- **Non-blocking sorts**: Parallel sorting never blocks rendering
- **Distance-based ordering**: Efficient camera rotation without re-sorting

## Quick Start
1. **Import splat assets** (PLY/SPZ) using the editor tooling
2. **Configure octree** settings (`maxDepth`, `maxSplatsPerLeaf`) - aim for ~1024 splats per leaf
3. **Enable Draw Gizmos** in settings to visualize octree bounds and tune parameters
4. **For WebGL**: Enable "Native C++ Threads Support" in Unity WebGL settings and serve with CORS headers (use included `serve.py`)

The system automatically builds an octree, filters outliers, and performs frustum culling and parallel sorting each frame.

**Try the example**: The GaussianExample-URP package includes a "Barangaroo" scene.

## WebGL Setup
For WebGL threading support (required for optimal performance):

1. **Unity Settings**: Enable "Native C++ Threads Support" in Project Settings > WebGL Settings
2. **Web Server**: Serve with CORS headers for SharedArrayBuffer support:
   ```bash
   python serve.py  # Use included script for local testing
   ```
   Or configure your server with:
   ```
   Cross-Origin-Opener-Policy: same-origin
   Cross-Origin-Embedder-Policy: require-corp
   ```
3. **Browser Support**: Modern browsers (Chrome 68+, Firefox 79+, Safari 15.2+)

Desktop platforms use Unity's Task system by default (no additional setup required).

## Configuration
| Setting | Default | Description |
|---------|---------|-------------|
| `maxDepth` | 8 | Octree depth (aim for ~1024 splats per leaf) |
| `maxSplatsPerLeaf` | 1024 | Maximum splats per octree leaf |
| `enableParallelSorting` | true | Enable multi-threaded sorting (fallback to sequential sort over frames when no thread support in the build) |
| `parallelSortThreads` | auto | Worker thread count (auto-detected)

## Performance Tips
- Start with `maxDepth = 8` and `maxSplatsPerLeaf = 1024` for most scenes
- Tune "scene splat ratio" (0.90-0.95) to filter distant noise into outlier bucket
- Check browser console for "SharedArrayBuffer is defined: true" on WebGL
- Optional: Compile native libraries for maximum desktop performance (see `docs/`)

## License
See `LICENSE.md`.
