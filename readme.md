# Unity Gaussian Splatting (WebGPU-Oriented Fork)

This fork pivots the original project toward maximum portability and future WebGPU build compatibility.
The goal: keep visual quality while removing features that block Web / constrained platforms, and replace
heavy global GPU sorting with algorithmic + structural reductions in work.

## Key Changes vs Original Project

1. Removed GPU radix sort dependency
   - Original device radix sort (compute shader) is not universally available (esp. early WebGPU / some mobile paths).
   - Replaced with hierarchical octree culling + localized (per-node) CPU sorting.

2. Hierarchical Octree Pipeline
   - Build-time spatial partition (depth configurable, max splats per leaf).
   - Frustum culling happens at node granularity first, drastically shrinking the candidate set.
   - Only visible leaf node splats are considered for ordering.
   - Outlier points (outside compact root bounds) are tracked separately and appended first (farthest group) after optional local sort.

3. Per-Node Threaded Sorting (Configurable)
   - Instead of one giant O(N log N) over all visible splats, we perform multiple small sorts.
   - Worker thread count configurable (`parallelSortThreads`). Main thread simultaneously sorts outlier group.
   - Uses ArrayPool-backed scratch buffers; no per-frame GC churn.
   - Falls back to single-thread seamlessly when too few nodes or single core environments (e.g. some Web builds without shared-array threads).

4. Time Complexity Reduction
   - Original global approach: O(V log V) where V = visible splats.
   - New approach: O(K log K_avg * S) where K = number of visible leaf nodes, K_avg = average splats per node, S = total visible splats.
     Typically K_avg << V leading to fewer comparisons and better cache locality.
   - Culling eliminates entire subtrees early; large invisible regions cost near O(1).
   - Parallelization further reduces wall clock when K large and K_avg moderately sized (sweet spot: many nodes of 64–512 splats).

5. Stochastic Non-Sort Experiment (Optional / WIP)
   - Idea: For dense regions with high overdraw, strict total ordering has diminishing returns.
   - Prototype path (not default): random jitter bucketting by coarse depth layer or reservoir-sampled subset ordering.
   - Beneficial for very large counts when alpha blending artifacts are visually masked by density.
   - Switch retained as an experimental toggle (not enabled by default) – documentation to follow when stabilized.

6. WebGPU Readiness
   - Avoids reliance on advanced compute shader intrinsics / barriers not uniformly present yet.
   - Uses CPU-based hierarchy and sorting so rendering path mainly needs standard raster pipeline + SSBO / structured buffer for indices.
   - Threaded sorting gracefully degrades: if WASM threads / SharedArrayBuffer are unavailable, it auto-falls back to single-thread.

7. Memory Footprint Improvements
   - Reuses single large scratch array (or pooled per-thread arrays) instead of allocating per sort call.
   - Octree nodes only store original indices; position lookup via compact array/dictionary.

## Current Workflow

1. Build / import a Gaussian Splat asset (PLY / SPZ as before).
2. Octree built once (configurable max depth & splats per leaf). Outliers separated.
3. Each frame (alpha blended mode):
   - Frustum cull nodes.
   - Collect visible leaf indices.
   - Optionally parallel per-node depth sort (configurable threads & threshold).
   - Concatenate ordered indices -> GPU buffer.
4. Renderer consumes already partially ordered list (back-to-front per node; nodes themselves pre-sorted by center distance).

## Why This Works

Rendering correctness for alpha blended splats ideally wants exact global back-to-front ordering. However:
- Spatial coherence: nearby splats have similar depths; perfect ordering inside a small cluster yields most visual benefit.
- Distant clusters rarely interpenetrate enough to cause dominant blending errors; cluster ordering by center is a good heuristic.
- Reduced sort domain + improved locality lowers L2 misses and branch mispredicts vs huge comparator chain.
- In practice (measured on multi-million splat scenes) visual parity with global sort while cutting sort time substantially.

## Configuration Flags (Runtime)

| Setting | Purpose |
|--------|---------|
| `maxDepth` / `maxSplatsPerLeaf` | Octree granularity trade-off (build time) |
| `enableParallelSorting` | Master toggle for multi-thread node sorting |
| `parallelSortThreads` | Desired worker threads (clamped to hardware & node count) |
| Experimental: `useStochasticOrdering` (future) | Activate non-deterministic coarse ordering path |

Recommended: Tune `maxSplatsPerLeaf` so average visible leaf count keeps per-node sorts meaningful (32–256 range).

## Limitations / Future

- Stochastic path still under evaluation (ghosting checks, temporal stability). Not default.
- Order Independent Transparency (weighted blended / dual depth peeling) intentionally avoided for Web portability.
- No global GPU radix fallback; if you target high-end native only, reintroducing compute sort may still be faster at extreme counts.

## Experimental Non-Sort Mode (Concept)

Instead of sorting every frame:
1. Bucket splats by quantized depth slice per node.
2. Shuffle within bucket or reuse cached order for several frames.
3. Occasionally re-evaluate when camera forward vector changes beyond threshold.
Goal: amortize ordering cost on static or slow-motion views.
(Status: design stage; not fully integrated.)

---

Happy splatting.
