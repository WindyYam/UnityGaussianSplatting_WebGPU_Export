# Unity Gaussian Splatting (WebGPU-Oriented Fork)

**Live WebGPU Demo:** https://windyyam.github.io/unity_splat_test_deploy/

A portability-focused fork aiming for high visual fidelity splat rendering that degrades gracefully on constrained / upcoming WebGPU platforms. Core idea: remove the monolithic global GPU radix sort and replace it with hierarchical culling + small, cache‑friendly per-node CPU sorts (optionally parallel) to keep ordering cost low.

## Highlights
- Hierarchical octree build (configurable depth & leaf capacity) + outlier bucket.
- Frustum culling at node granularity drastically reduces sort domain.
- Per-node (optionally parallel) depth sort using pooled buffers; auto fallback to single-thread (e.g. when WASM threads unavailable).
- Node centers pre-ordered; intra-node ordering yields near visual parity with full global sort for typical scenes.
- Experimental stochastic / amortized ordering path (design phase) for very dense scenes.
- WebGPU friendly: no reliance on advanced compute barriers or large global GPU radix passes.
- Memory conscious: reuses scratch buffers; nodes store indices only.

## Quick Start
1. Import / build a Gaussian Splat asset (PLY / SPZ) via the provided editor tooling.
2. Octree builds once (respect `maxDepth`, `maxSplatsPerLeaf`). Outliers separated automatically.
3. Each frame (alpha blend mode):
   - Frustum cull nodes.
   - Gather visible leaves.
   - (Optional) parallel per-leaf sort.
   - Concatenate indices -> single GPU buffer consumed by renderer.

## Runtime Settings
| Setting | Description | Guidance |
|---------|-------------|----------|
| `maxDepth`, `maxSplatsPerLeaf` | Octree granularity | Tune so typical visible leaf has ~32–256 splats |
| `enableParallelSorting` | Toggle multi-thread path | Leave on for desktop; harmless fallback on Web |
| `parallelSortThreads` | Requested worker threads | Clamped to hardware & visible leaf count |
| `useStochasticOrdering` (experimental) | Coarse / amortized ordering | Off by default; densest scenes only |

## Why Per-Node Sorting Works (Short Version)
Spatial coherence means most visual correctness comes from good local ordering. Distant clusters rarely interpenetrate; ordering nodes by center + sorting internally captures the bulk of alpha needs while slashing comparisons vs a global O(V log V) pass.

## Roadmap / Limitations
- Stochastic ordering: stabilize heuristics (temporal stability, ghosting checks).
- No OIT (weighted blended / dual depth peeling) to keep Web path lean.
- If you only target high-end native, reintroducing a global GPU sort could still win at extreme (> tens of millions) visible splat counts.

## Experimental Non-Sort Concept (In Design)
Bucket by quantized depth slice per node, shuffle or reuse order across frames, refresh when camera direction changes beyond threshold—amortizing cost for static views.

## Contributing
PRs welcome for: WebGPU backend improvements, WASM threading validation, stochastic ordering experiments, profiling harnesses.

## License
See `LICENSE.md`.

Happy splatting.
