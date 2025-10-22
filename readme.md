# Unity Gaussian Splatting (WebGPU Fork)

**Live Demo:** https://windyyam.github.io/unity_splat_test_deploy/

A WebGPU-focused fork of [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) optimized for high-fidelity Gaussian splat rendering on web platforms and constrained environments.

> Note on development: This repository originally followed the `stochastic` branch of the upstream project to investigate a GPU sort-free solution. Due to visible noise introduced by stochastic alpha blending, the project shifted to a CPU + octree-based sorting approach for more deterministic, high-fidelity results.

## Features
- **Cross-platform threading**: Native threading on WebGPU(C# runtime doesn't support threading yet on Unity), C# Tasks on desktop
- **Hierarchical octree** with configurable depth and outlier filtering
- **Frustum culling** reduces processing overhead
- **Non-blocking sorts**: Parallel sorting never blocks rendering
- **Distance-based ordering**: Efficient camera rotation without re-sorting compare to camera front based sorting
- **Cacheable node sort results**: Sort order for distant octree nodes is cached and reused across small camera movements, avoiding unnecessary re-sorts and improving frame performance. This is a key benefit of the approach — distant nodes whose relative order doesn't change significantly can use cached results to reduce CPU work.

## Quick Start
1. **Import splat assets** (PLY/SPZ) using the editor tooling
2. **Configure octree** settings (`maxDepth`, `maxSplatsPerLeaf`) - aim for depth ~6 and ~2048 splats per leaf
3. **Enable Draw Gizmos** in settings to visualize octree bounds and tune parameters
4. **For WebGPU**: Enable "Native C++ Threads Support" in Unity WebGPU settings and serve with CORS headers (use included `serve.py`)

The system automatically builds an octree, filters outliers, and performs frustum culling and parallel sorting each frame.

**Try the example**: The GaussianExample-URP package includes a "Barangaroo" scene.

## WebGPU Setup
For WebGPU threading support (required for optimal performance):

1. **Unity Settings**: Enable "Native C++ Threads Support" in Project Settings > WebGPU Settings
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

> Note: Even if you do not use C++/WASM threads, the sequential path is still remarkably good thanks to the cachable node sort result and distributed sort work over frames.

Desktop platforms use Unity's Task system by default (no additional setup required).

## Configuration
| Setting | Default | Description |
|---------|---------|-------------|
| `maxDepth` | 6 | Octree depth(to limit frustum AABB tests number) |
| `maxSplatsPerLeaf` | 2048 | Maximum splats per octree leaf (if not reached maxDepth yet) |

## Performance Tips
- Start with `maxDepth = 6` and `maxSplatsPerLeaf = 2048` for most scenes.
- For cases sensitive to node-wise distance sort error (large-node vs small-node ordering artifacts), prefer an even spatial split: set `maxSplatsPerLeaf = 1` to force uniform splitting down to `maxDepth`. This reduces inter-node ordering error at the cost of more leaves and higher CPU/memory overhead.
- Tune "scene splat ratio" (0.90-0.95) to filter distant noise into outlier bucket
- Check browser console for "WebGL platform — using native threading" on WebGPU
- Optional: Compile native libraries for maximum desktop performance (see `docs/`)

## License
See `LICENSE.md`.
