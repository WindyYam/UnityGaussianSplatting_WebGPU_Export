# Unity Gaussian Splatting (WebGPU-Oriented Fork)

**Live WebGPU Demo:** https://windyyam.github.io/unity_splat_test_deploy/

This repository is a fork of `aras-p/UnityGaussianSplatting`. Many thanks to @aras-p for the original project and the work that made this fork possible: https://github.com/aras-p/UnityGaussianSplatting

This fork originally followed the `stochastic` branch of the original project and experimented with stochastic sort-free alpha blend. It was heavily modified to build on WebGPU — including removal of the compute shader path — but the stochastic approach introduced noise that degrades the visual quality. As a result, the implementation reverted to CPU-side per-node sorting, augmented with node culling, caching of per-node orderings, and distance-based ordering to mitigate cost and visual artifacts.

As a result, this is a portability-focused fork primarily targeting WebGPU with high visual fidelity splat rendering that works well on constrained / upcoming WebGPU platforms. To maximize compatibility on Web and WASM, the implementation favors CPU-side hierarchical culling and small per-node sorts (which run in parallel when available) rather than relying on large, complex GPU-only global sorts.

Edit: Not until later I realise WASM thread is not available in Unity C# at the moment, so parellel sort doesn't work in WebGPU at all(sad), then why the heck it looks visually fast? I've no idea to to this, maybe the sort overhead is much less than I'd thought? Or the cache node optimization is very good here?
The only path for parelleling right now is experimental native C++ thread support, which is under development.

## Highlights
- Hierarchical octree build (configurable depth & leaf capacity) + outlier bucket.
- Frustum culling at node granularity drastically reduces sort domain.
- Per-node sorts run asynchronously in the background (non‑blocking), updating cached orderings when complete so rendering is not blocked by sorting.
- Node centers are pre-ordered by distance to the camera position, and individual splats inside each node are also sorted by distance (not by camera forward). This preserves the sort order when the camera rotates in place, no re-sort needed as opposite to cam forward based sorting.
- WebGPU-first: CPU-friendly algorithms ensure correctness and portability where advanced GPU features or threading are limited.

## Quick Start
1. Import / build a Gaussian Splat asset (PLY / SPZ) via the provided editor tooling.
2. Octree builds once (respect `maxDepth`, `maxSplatsPerLeaf`). Outliers separated automatically during preprocessing: very far-away sparse splats (e.g. sky pixels, distant noise, or background samples produced by MCMC strategies) are detected and placed into an outlier bucket and treated as background. This lets the octree build focus on the main dense splat region — improving node efficiency and reducing per-node sort cost. The aggressiveness of the outlier filter can be tuned via the "scene splat ratio" setting to control how many splats are included in the main octree build versus moved to the outlier bucket.

   Tip: You can visualize the octree node bounds and leaf centers in the Unity Scene view by enabling the "Draw Gizmos" checkbox in the Gaussian Splat settings script. This overlay is useful for diagnosing outliers and tuning `maxDepth`, `maxSplatsPerLeaf`, and the scene splat ratio.
   3. Each frame (alpha blend mode):
   - Frustum cull nodes.
   - Gather visible leaves.
   - Sort visible leaves (by node center distance, not precise, but until we find better solution).
   - Kick off parallel per-leaf sorts in the background (non‑blocking) or main thread depending on platform. Per-leaf sorting runs asynchronously and updates per-node cached orderings when complete.
   - Concatenate currently available per-node indices by leaves sort order into a single GPU buffer consumed by the renderer; updated orderings are picked up on subsequent frames once background sorts finish.

Note: The GaussianExample-URP package includes a ready-to-play scene named "Barangaroo".

## Runtime Settings
| Setting | Description | Guidance |
|---------|-------------|----------|
| `maxDepth`, `maxSplatsPerLeaf` | Octree granularity | Tune so typical visible leaf has ~1024 splats |
| `enableParallelSorting` | Toggle multi-thread path | Put it on, will be automatically disabled in Web platform |
| `parallelSortThreads` | Requested worker threads | Default 8 and will be set to hardware number |

Recommended defaults and guidance:
- Keep node sizes reasonable so per-leaf sorts stay cheap. A good starting point for many scenes is `maxDepth = 8` and `maxSplatsPerLeaf = 1024`.
- For the outlier filter (scene splat ratio), try `0.90`–`0.95` to push very sparse, far-away splats into the outlier bucket; if your scene is very clean (no distant noise) you can set this to `1.0`.

## Roadmap / Limitations
- If you only target high-end native, reintroducing a global GPU radix sort could still win at extreme (> tens of millions) visible splat counts.
- The visible-node sort used at runtime is an approximation for splat ordering and may need a more robust artifact-free solution.

## License
See `LICENSE.md`.

Happy splatting.
