// SPDX-License-Identifier: MIT

using System; // Added for Exception
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
// Added for simple threading support
using System.Threading;
using System.Buffers;
using System.Threading.Tasks;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Octree-based spatial acceleration structure for Gaussian splat frustum culling.
    /// Divides scene bounds into hierarchical octants for efficient culling of static splats.
    /// </summary>
    public class GaussianSplatOctree
    {
        public class OctreeNode
        {
            public Bounds bounds;
            public Vector3 center;
            // For leaf nodes we store original splat indices that lie within this node's bounds.
            // For internal nodes this may be null or empty, will be sorted.
            public List<int> splatIndices;
            // Child node indices (indices into m_Nodes). Null or empty for leaf nodes.
            public List<int> childIndices;
            public bool isLeaf;
            // Track if this node's splats are sorted for current camera view
            public bool isSorted;
            // Store the camera position used for last sort (to detect when re-sort needed)
            public Vector3 lastSortCameraPosition;
            // Cached maximum extent (largest half-size axis) for angular sort threshold calculations
            public float maxExtent;
            // Persistent native copy of splat indices for native sorting (read-only input).
            public NativeArray<int> nativeSplatIndices;
            public bool nativeIndicesValid;
        }

        public struct SplatInfo
        {
            public float3 position;
            public int originalIndex;
        }

        readonly List<OctreeNode> m_Nodes = new();
        NativeArray<int> m_VisibleSplatIndices;
        bool m_VisibleSplatIndicesValid;
        int m_TotalSplats; // Persist total splat count after releasing build-time list

        // Configuration
        int m_MaxDepth;
        int m_MaxSplatsPerLeaf;
        Bounds m_RootBounds;
        bool m_Built;

        // GPU buffer for visible splat indices (updated per frame/N frames)
        GraphicsBuffer m_VisibleIndicesBuffer;

        // Outlier splat indices that lie outside the main root bounds (always included in culling)
        readonly List<int> m_OthersIndices = new();
        // Persistent native copy of outlier indices (read-only input for native sort). Lazy-init.
        NativeArray<int> m_OthersNativeIndices;
        bool m_OthersNativeValid;
        
        // Reusable array for distance sorting to avoid allocations
        (float distance, int index)[] m_DistanceSortArray;

        // Structure to store visible node references with their distance for hierarchical sorting
        struct VisibleNodeRef
        {
            public float distance;
            public int nodeIndex; // Index into m_Nodes instead of copying splat indices
        }

        // Reusable list for visible node references during sorting
        readonly List<VisibleNodeRef> m_VisibleNodeRefs = new();
        
        // Reusable stack for non-recursive octree traversal
        readonly Stack<int> m_TraversalStack = new();
        
        // Enable / disable parallel sorting (public for runtime tuning)
        public bool enableParallelSorting = true;
        // Configurable number of worker threads for node sorting (excluding main thread)
        public int parallelSortThreads = 8; // Default sort threads, Safe for most of the platform
        // Maximum number of nodes to sort per frame to ensure closest nodes are prioritized during camera movement
        // This prevents frame time spikes by limiting sort work and ensures closest nodes are sorted first
        public int maxSortNodesPerFrame = 256; // In sequential path we do sort over time
        // Angular threshold for re-sorting: minimum cosine of angle change before re-sort is needed
        // cosine(15°) ≈ 0.966, cosine(30°) ≈ 0.866, cosine(45°) ≈ 0.707
        public float sortDirectionThreshold = 0.9f; // ~25.8° angle change threshold
        // Simplified outlier sorting strategy:
        // We keep an average radial distance (ring radius) of outliers from the root center.
        // Re-sorting occurs only when camera has moved more than (outlierRingRadius * outlierResortMoveFraction).
        // If the computed ring radius is zero (edge case), we fall back to a small constant.
        public float outlierResortMoveFraction = 0.1f; // 10% of ring radius movement triggers re-sort
        public float minOutlierResortDistance = 0.05f; // Fallback minimum distance if radius very small
        bool m_OthersSorted;            // Track if outliers are currently sorted
        Vector3 m_LastOthersSortCamPos; // Camera position at last outlier sort
        float m_OutlierRingRadius;      // Average radial distance of outliers from scene center

        Task[] m_SortTasks;
        
        // Native sorting job handles for WebGL platform
        readonly List<NativeSorting.SortJobHandle> m_NativeSortJobs = new();
        // Track which jobs correspond to which data structures
        readonly List<NativeSortJobInfo> m_NativeJobInfos = new();

        // Global native positions buffer (all splat positions) to avoid per-job copying
        NativeArray<float3> m_AllPositionsNative;
        bool m_AllPositionsNativeValid;
        
        struct NativeSortJobInfo
        {
            public bool isOutlierJob;
            public int nodeIndex; // -1 for outlier jobs
            public Vector3 cameraPosition;
            public NativeArray<int> inputIndices; // Per-job input splat indices (owned by job if disposeInput == true)
            public NativeArray<int> sortedIndices; // Per-job output sorted indices
            public bool disposeInput; // Whether we should dispose inputIndices when job completes
        }

        public int nodeCount => m_Nodes.Count;
        public int totalSplats => m_TotalSplats;
        public bool isBuilt => m_Built;
        public GraphicsBuffer visibleIndicesBuffer => m_VisibleIndicesBuffer;
        public int visibleSplatCount { get; private set; }

        // Helper to get splat position directly from global native buffer
        bool TryGetSplatPosition(int originalIndex, out float3 pos)
        {
            if (m_AllPositionsNativeValid && originalIndex >= 0 && originalIndex < m_AllPositionsNative.Length)
            {
                pos = m_AllPositionsNative[originalIndex];
                return true;
            }
            pos = default;
            return false;
        }

        // Helper to ensure the visible splat indices native array is large enough
        void EnsureVisibleSplatIndicesCapacity(int requiredCapacity)
        {
            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated || m_VisibleSplatIndices.Length < requiredCapacity)
            {
                if (m_VisibleSplatIndicesValid && m_VisibleSplatIndices.IsCreated)
                {
                    try { m_VisibleSplatIndices.Dispose(); } catch {}
                }
                
                // Allocate with some extra space to avoid frequent reallocations
                int bufferSize = Mathf.NextPowerOfTwo(Mathf.Max(requiredCapacity, 1));
                try
                {
                    m_VisibleSplatIndices = new NativeArray<int>(bufferSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    m_VisibleSplatIndicesValid = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Failed to allocate visible splat indices native array: {ex.Message}");
                    m_VisibleSplatIndicesValid = false;
                }
            }
        }

        /// <summary>
        /// Initialize octree parameters. Call this before building.
        /// </summary>
        /// <param name="maxDepth">Maximum tree depth (typically 4-6)</param>
        /// <param name="maxSplatsPerLeaf">Maximum splats per leaf node (typically 64-256)</param>
        public void Initialize(int maxDepth = 5, int maxSplatsPerLeaf = 128)
        {
            m_MaxDepth = maxDepth;
            m_MaxSplatsPerLeaf = maxSplatsPerLeaf;

            // Print available system cores (both .NET and Unity reports)
            int envCores = Environment.ProcessorCount;
            int unityCores = SystemInfo.processorCount;
            Debug.Log($"Available cores - Environment.ProcessorCount: {envCores}, SystemInfo.processorCount: {unityCores} (Might not be accurate on Web platform)");

            // Check if native threading is supported on current platform
            bool isWebPlatform = Application.platform == RuntimePlatform.WebGLPlayer;
            
            if(enableParallelSorting) 
            {
                // Try to initialize native sorting for supported platforms
                int nativeWorkers = Mathf.Max(1, envCores - 1); // Conservative worker count for all platforms
                NativeSorting.Initialize(nativeWorkers);

                int nativeWorkerCount = NativeSorting.GetWorkerCount();
                if (NativeSorting.IsAvailable && nativeWorkerCount > 0)
                {
                    parallelSortThreads = nativeWorkerCount;

                    string platformName = isWebPlatform ? "WebGL" : "native";
                    Debug.Log($"GaussianSplatOctree: {platformName} platform — using native threading with {parallelSortThreads} workers");
                }
                else if (isWebPlatform)
                {
                    // WebGL without native support - fallback to sequential
                    enableParallelSorting = false;
                    Debug.LogWarning("GaussianSplatOctree: WebGL platform — native threading unavailable, using single-threaded fallback.");
                }
                else
                {
                    int reportedCores = SystemInfo.processorCount;
                    if (reportedCores > 0)
                        parallelSortThreads = reportedCores;
                }
            }
            
            if(enableParallelSorting)
                // Inform about the number of threads that will be used for parallel sorting
                Debug.Log($"GaussianSplatOctree: parallelSortThreads set to {parallelSortThreads}");
        }

        /// <summary>
        /// Build octree from splat position data and bounds.
        /// </summary>
        public void Build(NativeArray<float3> splatPositions, Bounds sceneBounds, float splatPercent)
        {
            Clear();
            // m_OthersNodeIndex removed - use m_OthersIndices list instead

            if (splatPositions.Length == 0)
            {
                Debug.LogWarning("GaussianSplatOctree.Build: No splat positions provided");
                return;
            }

            Debug.Log($"Building octree with {splatPositions.Length} splats, bounds: {sceneBounds}");

            // Compute center of mass and identify 95% closest splats
            int total = splatPositions.Length;
            m_TotalSplats = total;
            float3 com = float3.zero;
            for (int i = 0; i < total; i++)
                com += splatPositions[i];
            com /= total;

            var distList = new List<(int idx, float d)>(total);
            for (int i = 0; i < total; i++)
            {
                float distance = math.distance(splatPositions[i], com);
                distList.Add((i, distance));
            }
            distList.Sort((a, b) => a.d.CompareTo(b.d));

            // Reorder m_SplatInfos so that the closest part are first, others last
            int inCount = Mathf.CeilToInt(total * splatPercent);
            inCount = Mathf.Clamp(inCount, 1, total);
            int othersCount = total - inCount;

            // Local build-time splat info list
            var splatInfos = new List<SplatInfo>(total);
            for (int i = 0; i < total; i++)
            {
                int src = distList[i].idx;
                splatInfos.Add(new SplatInfo { position = splatPositions[src], originalIndex = src });
            }

            // Create / update global native positions buffer
            if (m_AllPositionsNativeValid)
            {
                if (m_AllPositionsNative.IsCreated) m_AllPositionsNative.Dispose();
                m_AllPositionsNativeValid = false;
            }
            try
            {
                m_AllPositionsNative = new NativeArray<float3>(total, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < splatInfos.Count; i++)
                {
                    var si = splatInfos[i];
                    int orig = si.originalIndex;
                    if ((uint)orig < (uint)total)
                        m_AllPositionsNative[orig] = si.position;
                }
                m_AllPositionsNativeValid = true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Failed to allocate global native positions buffer: {ex.Message}");
                if (m_AllPositionsNative.IsCreated) m_AllPositionsNative.Dispose();
                m_AllPositionsNativeValid = false;
            }

            // Create root bounds based on the inCount splats (centered on center-of-mass)
            Bounds rootBounds;
            if (inCount > 0)
            {
                float3 min = splatInfos[0].position;
                float3 max = splatInfos[0].position;
                for (int i = 1; i < inCount; i++)
                {
                    min = math.min(min, splatInfos[i].position);
                    max = math.max(max, splatInfos[i].position);
                }
                rootBounds = new Bounds((max + min) * 0.5f, max - min);
            }
            else
            {
                // Fallback to provided scene bounds
                rootBounds = sceneBounds;
            }

            m_RootBounds = rootBounds;

            // Build tree recursively using only the in-root splats
            m_Nodes.Clear();

            // Create root node covering the in-root splats
            var rootNode = new OctreeNode
            {
                bounds = m_RootBounds,
                center = m_RootBounds.center,
                splatIndices = null,
                childIndices = null,
                isLeaf = false,
                maxExtent = Mathf.Max(m_RootBounds.extents.x, Mathf.Max(m_RootBounds.extents.y, m_RootBounds.extents.z))
            };
            m_Nodes.Add(rootNode);

            // Build recursively starting from root (only for the in-root partition)
            var rootSplatList = new List<int>(inCount);
            for (int i = 0; i < inCount; i++) rootSplatList.Add(i); // indices into splatInfos
            BuildRecursive(0, 0, rootSplatList, splatInfos);

            // Handle remaining outliers: put their original indices into m_SplatIndices and track them in m_OthersIndices
            m_OthersIndices.Clear();
            if (othersCount > 0)
            {
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = splatInfos[inCount + i].originalIndex;
                    m_OthersIndices.Add(orig);
                }
            }
            m_OthersSorted = false; // reset outlier sorting state after build
            m_LastOthersSortCamPos = Vector3.zero;
            // Compute average outlier ring radius (ignore min/max & extra stats for simplicity)
            m_OutlierRingRadius = 0f;
            if (othersCount > 0 && m_AllPositionsNativeValid)
            {
                Vector3 center = m_RootBounds.center;
                double accum = 0.0;
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = splatInfos[inCount + i].originalIndex;
                    if (orig >= 0 && orig < m_AllPositionsNative.Length)
                    {
                        float3 p = m_AllPositionsNative[orig];
                        accum += Vector3.Distance(center, (Vector3)p);
                    }
                }
                m_OutlierRingRadius = (float)(accum / othersCount);
            }

            // Tighten bounding boxes starting from leaves and propagating up
            TightenBounds();

            m_Built = true;

            // Ensure a GPU buffer exists even if there are no visible splats yet.
            // Allocate a minimal 1-entry structured buffer so renderer code can safely bind/check it.
            if (m_VisibleIndicesBuffer == null)
            {
                m_VisibleIndicesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint))
                {
                    name = "GaussianSplatVisibleIndices"
                };
                var init = new NativeArray<uint>(1, Allocator.Temp);
                init[0] = 0u;
                m_VisibleIndicesBuffer.SetData(init);
                visibleSplatCount = 0;
                init.Dispose();
            }

            Debug.Log($"Octree build completed: {m_Nodes.Count} total nodes, others={m_OthersIndices.Count}");

            EnsureVisibleSplatIndicesCapacity(m_TotalSplats);
        }

        void BuildRecursive(int nodeIndex, int depth, List<int> splatList, List<SplatInfo> splatInfos)
        {
            var node = m_Nodes[nodeIndex];

            // Check termination conditions
            if (depth >= m_MaxDepth || splatList.Count <= m_MaxSplatsPerLeaf)
            {
                // Make this a leaf node and store original indices for this leaf
                node.isLeaf = true;
                node.splatIndices = new List<int>(splatList.Count);
                for (int i = 0; i < splatList.Count; i++)
                {
                    int infoIdx = splatList[i];
                    if (infoIdx < 0 || infoIdx >= splatInfos.Count)
                    {
                        Debug.LogError($"Octree leaf node splat info index out of bounds: {infoIdx} >= {splatInfos.Count}");
                        continue;
                    }
                    node.splatIndices.Add(splatInfos[infoIdx].originalIndex);
                }

                m_Nodes[nodeIndex] = node;
                return;
            }

            // Create 8 child nodes
            var center = node.bounds.center;
            var size = node.bounds.size * 0.5f;

            node.childIndices = new List<int>(8);
            node.isLeaf = false;
            m_Nodes[nodeIndex] = node;

            // Create child bounds
            var childBounds = new Bounds[8];
            for (int i = 0; i < 8; i++)
            {
                var offset = new Vector3(
                    (i & 1) != 0 ? size.x * 0.5f : -size.x * 0.5f,
                    (i & 2) != 0 ? size.y * 0.5f : -size.y * 0.5f,
                    (i & 4) != 0 ? size.z * 0.5f : -size.z * 0.5f
                );
                childBounds[i] = new Bounds(center + offset, size);
            }

            // Distribute splats to children
            var childSplatsIdx = new List<int>[8];
            for (int i = 0; i < 8; i++) childSplatsIdx[i] = new List<int>();

            // Assign splats (using splatList which holds indices into m_SplatInfos) to child nodes
            for (int ii = 0; ii < splatList.Count; ii++)
            {
                int infoIdx = splatList[ii];
                if (infoIdx < 0 || infoIdx >= splatInfos.Count)
                {
                    Debug.LogError($"Octree splat distribution info index out of bounds: {infoIdx} >= {splatInfos.Count}");
                    continue;
                }

                var splat = splatInfos[infoIdx];

                int childIndex = 0;
                if (splat.position.x > center.x) childIndex |= 1;
                if (splat.position.y > center.y) childIndex |= 2;
                if (splat.position.z > center.z) childIndex |= 4;

                childSplatsIdx[childIndex].Add(infoIdx);
            }

            // Create child nodes
            for (int i = 0; i < 8; i++)
            {
                var childNode = new OctreeNode
                {
                    bounds = childBounds[i],
                    center = childBounds[i].center,
                    splatIndices = null,
                    childIndices = null,
                    isLeaf = childSplatsIdx[i].Count == 0,
                    maxExtent = Mathf.Max(childBounds[i].extents.x, Mathf.Max(childBounds[i].extents.y, childBounds[i].extents.z))
                };

                int childNodeIndex = m_Nodes.Count;
                m_Nodes.Add(childNode);

                // Register child index with parent
                node.childIndices.Add(childNodeIndex);
                // Update parent reference in the global list (node is a reference type)
                m_Nodes[nodeIndex] = node;

                // Recursively build child only if it has splats
                if (childSplatsIdx[i].Count > 0)
                {
                    BuildRecursive(childNodeIndex, depth + 1, childSplatsIdx[i], splatInfos);
                }
            }
        }

        /// <summary>
        /// Tighten bounding boxes for all nodes based on actual splat positions.
        /// Starts from leaf nodes and propagates up to parent nodes.
        /// </summary>
        void TightenBounds()
        {
            if (m_Nodes.Count == 0)
                return;

            int tightenedNodes = 0;
            
            // Process nodes in reverse order to handle leaves first, then propagate up
            for (int i = m_Nodes.Count - 1; i >= 0; i--)
            {
                if (TightenNodeBounds(i))
                    tightenedNodes++;
            }

            Debug.Log($"Octree bounds tightened: {tightenedNodes}/{m_Nodes.Count} nodes updated");
        }

        /// <summary>
        /// Tighten the bounds of a specific node based on its splats or child bounds.
        /// </summary>
        /// <returns>True if the bounds were changed, false otherwise</returns>
        bool TightenNodeBounds(int nodeIndex)
        {
            if (nodeIndex >= m_Nodes.Count)
                return false;

            var node = m_Nodes[nodeIndex];
            var originalBounds = node.bounds;

            if (node.isLeaf)
            {
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    int firstSplatIdx = node.splatIndices[0];
                    if (TryGetSplatPosition(firstSplatIdx, out float3 firstPos))
                    {
                        float3 min = firstPos;
                        float3 max = firstPos;
                        for (int i = 1; i < node.splatIndices.Count; i++)
                        {
                            int splatIdx = node.splatIndices[i];
                            if (TryGetSplatPosition(splatIdx, out float3 pos))
                            {
                                min = math.min(min, pos);
                                max = math.max(max, pos);
                            }
                        }
                        Vector3 center = (Vector3)((min + max) * 0.5f);
                        Vector3 size = (Vector3)(max - min);
                        const float minSize = 0.001f;
                        size.x = Mathf.Max(size.x, minSize);
                        size.y = Mathf.Max(size.y, minSize);
                        size.z = Mathf.Max(size.z, minSize);
                        node.bounds = new Bounds(center, size);
                        node.maxExtent = Mathf.Max(size.x, Mathf.Max(size.y, size.z)) * 0.5f;
                        m_Nodes[nodeIndex] = node;
                        return !BoundsAreEqual(originalBounds, node.bounds);
                    }
                }
                return false;
            }
            else
            {
                // For internal nodes, calculate bounds based on child node bounds
                if (node.childIndices != null && node.childIndices.Count > 0)
                {
                    bool hasValidChild = false;
                    float3 min = float3.zero;
                    float3 max = float3.zero;

                    foreach (int childIndex in node.childIndices)
                    {
                        if (childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];
                            
                            // Only include non-empty children in bounds calculation
                            bool childHasContent = childNode.isLeaf 
                                ? (childNode.splatIndices != null && childNode.splatIndices.Count > 0)
                                : (childNode.childIndices != null && childNode.childIndices.Count > 0);

                            if (childHasContent)
                            {
                                Vector3 childMin = childNode.bounds.min;
                                Vector3 childMax = childNode.bounds.max;

                                if (!hasValidChild)
                                {
                                    min = (float3)childMin;
                                    max = (float3)childMax;
                                    hasValidChild = true;
                                }
                                else
                                {
                                    min = math.min(min, (float3)childMin);
                                    max = math.max(max, (float3)childMax);
                                }
                            }
                        }
                    }

                    // Update bounds if we found valid children
                    if (hasValidChild)
                    {
                        Vector3 center = (Vector3)((min + max) * 0.5f);
                        Vector3 size = (Vector3)(max - min);
                        
                        // Ensure minimum size to avoid zero-size bounds
                        const float minSize = 0.001f;
                        size.x = Mathf.Max(size.x, minSize);
                        size.y = Mathf.Max(size.y, minSize);
                        size.z = Mathf.Max(size.z, minSize);

                        node.bounds = new Bounds(center, size);
                        node.maxExtent = Mathf.Max(size.x, Mathf.Max(size.y, size.z)) * 0.5f;

                        // Update the node in the list
                        m_Nodes[nodeIndex] = node;
                        
                        // Check if bounds actually changed
                        return !BoundsAreEqual(originalBounds, node.bounds);
                    }
                }
                return false; // No valid children, bounds unchanged
            }
        }

        /// <summary>
        /// Helper method to compare two bounds for equality with small tolerance.
        /// </summary>
        bool BoundsAreEqual(Bounds a, Bounds b)
        {
            const float tolerance = 1e-6f;
            return Vector3.Distance(a.center, b.center) < tolerance && 
                   Vector3.Distance(a.size, b.size) < tolerance;
        }

        /// <summary>
        /// Perform frustum culling and update visible splat indices.
        /// Returns number of visible splats.
        /// </summary>
        public int CullFrustum(Camera camera)
        {
            if (!m_Built)
                return 0;

            // Estimate capacity needed (total splats as upper bound)
            int estimatedCapacity = m_TotalSplats;

            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
            {
                visibleSplatCount = 0;
                return 0;
            }

            // Extract frustum planes from camera
            var frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);

            // Traverse octree and collect visible splats
            int currentIndex = 0;
            CullNodeRecursive(0, frustumPlanes, ref currentIndex);

            // Always include 'others' outlier splats
            if (m_OthersIndices.Count > 0)
            {
                // Ensure we have enough space for outliers
                if (currentIndex + m_OthersIndices.Count > m_VisibleSplatIndices.Length)
                {
                    if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
                    {
                        visibleSplatCount = 0;
                        return 0;
                    }
                }
                
                // Copy outlier indices
                for (int i = 0; i < m_OthersIndices.Count; i++)
                {
                    m_VisibleSplatIndices[currentIndex + i] = m_OthersIndices[i];
                }
                currentIndex += m_OthersIndices.Count;
            }

            visibleSplatCount = currentIndex;

            // Update GPU buffer
            UpdateVisibleIndicesBuffer();

            return visibleSplatCount;
        }

        void CullNodeRecursive(int nodeIndex, Plane[] frustumPlanes, ref int currentIndex)
        {
            if (nodeIndex >= m_Nodes.Count)
                return;

            var node = m_Nodes[nodeIndex];

            // Test node bounds against frustum
            if (!GeometryUtility.TestPlanesAABB(frustumPlanes, node.bounds))
                return; // Node is outside frustum

            if (node.isLeaf)
            {
                // Add all splats in this leaf to visible list (skip empty leaves)
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    // Ensure we have enough space
                    if (currentIndex + node.splatIndices.Count > m_VisibleSplatIndices.Length)
                    {
                        if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
                            return;
                    }
                    
                    // Copy splat indices
                    for (int i = 0; i < node.splatIndices.Count; i++)
                    {
                        m_VisibleSplatIndices[currentIndex + i] = node.splatIndices[i];
                    }
                    currentIndex += node.splatIndices.Count;
                }
            }
            else
            {
                // Recursively test child nodes - only traverse non-empty children
                // Traverse registered child indices
                if (node.childIndices != null)
                {
                    foreach (var childIndex in node.childIndices)
                    {
                        if (childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];
                            if ((childNode.splatIndices != null && childNode.splatIndices.Count > 0) || !childNode.isLeaf)
                            {
                                CullNodeRecursive(childIndex, frustumPlanes, ref currentIndex);
                            }
                        }
                    }
                }
            }
        }

        void UpdateVisibleIndicesBuffer()
        {
            if (visibleSplatCount == 0)
                return;

            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
            {
                Debug.LogWarning("Visible splat indices native array is invalid during buffer update");
                return;
            }

            // Ensure buffer is large enough
            int requiredSize = visibleSplatCount;
            if (m_VisibleIndicesBuffer == null || m_VisibleIndicesBuffer.count < requiredSize)
            {
                m_VisibleIndicesBuffer?.Dispose();
                // Allocate with some extra space to avoid frequent reallocations
                int bufferSize = Mathf.NextPowerOfTwo(requiredSize);
                m_VisibleIndicesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, sizeof(uint))
                {
                    name = "GaussianSplatVisibleIndices"
                };
            }

            // Upload visible indices directly from native array (reinterpret cast from int to uint)
            unsafe
            {
                // Create a NativeArray<uint> view of our int data (reinterpret cast)
                var uintView = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<uint>(
                    (void*)m_VisibleSplatIndices.GetUnsafeReadOnlyPtr(),
                    visibleSplatCount,
                    Allocator.None);
                
                #if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref uintView, 
                    NativeArrayUnsafeUtility.GetAtomicSafetyHandle(m_VisibleSplatIndices));
                #endif
                
                m_VisibleIndicesBuffer.SetData(uintView, 0, 0, visibleSplatCount);
            }
        }

        /// <summary>
        /// Get debug information about octree structure.
        /// </summary>
        public void GetDebugInfo(out int leafNodes, out int maxDepthReached, out int maxSplatsInLeaf)
        {
            leafNodes = 0;
            maxDepthReached = 0;
            maxSplatsInLeaf = 0;

            GetDebugInfoRecursive(0, 0, ref leafNodes, ref maxDepthReached, ref maxSplatsInLeaf);
        }

        void GetDebugInfoRecursive(int nodeIndex, int depth, ref int leafNodes, ref int maxDepth, ref int maxSplats)
        {
            if (nodeIndex >= m_Nodes.Count)
                return;

            var node = m_Nodes[nodeIndex];
            maxDepth = Mathf.Max(maxDepth, depth);

            if (node.isLeaf)
            {
                // Only count non-empty leaves
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    leafNodes++;
                    maxSplats = Mathf.Max(maxSplats, node.splatIndices.Count);
                }
            }
            else
            {
                // Traverse registered child indices
                if (node.childIndices != null)
                {
                    foreach (var childIndex in node.childIndices)
                    {
                        if (childIndex < m_Nodes.Count)
                        {
                            GetDebugInfoRecursive(childIndex, depth + 1, ref leafNodes, ref maxDepth, ref maxSplats);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Draw wireframe boxes for each non-empty leaf node. Call this from a MonoBehaviour's OnDrawGizmos or OnDrawGizmosSelected.
        /// </summary>
        public void DrawLeafBoundsGizmos(Color color)
        {
            if (!m_Built || m_Nodes.Count == 0)
                return;

            var prev = Gizmos.color;
            Gizmos.color = color;

            for (int i = 0; i < m_Nodes.Count; i++)
            {
                var node = m_Nodes[i];
                if (!node.isLeaf)
                    continue;

                // Skip empty leaves
                if (node.splatIndices == null || node.splatIndices.Count <= 0)
                    continue;

                Gizmos.DrawWireCube(node.bounds.center, node.bounds.size);
            }

            Gizmos.color = prev;
        }

        public void Clear()
        {
            // Cleanup native sorting jobs
            CleanupNativeSortJobs();
            // Dispose per-node native buffers before clearing list
            for (int i = 0; i < m_Nodes.Count; i++)
            {
                var n = m_Nodes[i];
                if (n.nativeIndicesValid && n.nativeSplatIndices.IsCreated)
                {
                    try { n.nativeSplatIndices.Dispose(); } catch {}
                    n.nativeIndicesValid = false;
                }
            }
            if (m_OthersNativeValid && m_OthersNativeIndices.IsCreated)
            {
                try { m_OthersNativeIndices.Dispose(); } catch {}
                m_OthersNativeValid = false;
            }
            m_Nodes.Clear();
            if (m_VisibleSplatIndicesValid && m_VisibleSplatIndices.IsCreated)
            {
                try { m_VisibleSplatIndices.Dispose(); } catch {}
                m_VisibleSplatIndicesValid = false;
            }
            m_VisibleNodeRefs.Clear();
            m_TraversalStack.Clear(); // Clear the reusable stack
            m_VisibleIndicesBuffer?.Dispose();
            m_VisibleIndicesBuffer = null;
            m_DistanceSortArray = null; // Release sort array memory
            visibleSplatCount = 0;
            m_Built = false;
            m_OthersIndices.Clear();
            m_OthersSorted = false;
            m_LastOthersSortCamPos = Vector3.zero;
            m_OutlierRingRadius = 0f;

            if (m_AllPositionsNativeValid && m_AllPositionsNative.IsCreated)
            {
                m_AllPositionsNative.Dispose();
                m_AllPositionsNativeValid = false;
            }

            m_TotalSplats = 0;
        }

        public void Dispose()
        {
            Clear();
            
            // Shutdown native sorting if it was initialized
            if (NativeSorting.IsAvailable)
            {
                NativeSorting.Shutdown();
            }
        }
        
        void CleanupNativeSortJobs()
        {
            for (int i = m_NativeSortJobs.Count - 1; i >= 0; i--)
            {
                var handle = m_NativeSortJobs[i];
                if (handle.IsValid)
                {
                    try { NativeSorting.CleanupJob(handle); } catch { }
                }
                var info = m_NativeJobInfos[i];
                try
                {
                    if (info.disposeInput && info.inputIndices.IsCreated) info.inputIndices.Dispose();
                    if (info.sortedIndices.IsCreated) info.sortedIndices.Dispose();
                }
                catch { }
                m_NativeSortJobs.RemoveAt(i);
                m_NativeJobInfos.RemoveAt(i);
            }
        }

        /// <summary>
        /// Sort visible splat indices by 3D distance from camera (front-to-back for alpha blending).
        /// Hierarchical sorting optimization.
        /// </summary>
        public void SortVisibleSplatsByDepth(Camera camera)
        {
            if (!m_Built)
                return;
            var camPosition = camera.transform.position;
            
            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
            {
                visibleSplatCount = 0;
                return;
            }
            
            m_VisibleNodeRefs.Clear();
            var frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);
            CollectVisibleNodesWithDistance(0, frustumPlanes, camPosition);

            if (enableParallelSorting)
            {
                if (NativeSorting.IsAvailable)
                {
                    // Use native sorting for supported platforms
                    // Collect results from any completed jobs and check if all are finished (non-blocking)
                    bool previousNativeJobsCompleted = CollectNativeSortResults();
                    
                    // Start new native sort jobs only if previous ones are completed
                    if (previousNativeJobsCompleted)
                    {
                        int jobsStarted = StartNativeSortJobs(camPosition);
                        //if (jobsStarted > 0)
                        //    Debug.Log($"Started {jobsStarted} new native sort jobs");
                    }
                    
                    // Sort node references by distance while native work happens in background
                    m_VisibleNodeRefs.Sort((a, b) => a.distance.CompareTo(b.distance)); // Front-to-back
                }
                else
                {
                    // Use Unity Task system for other platforms
                    // Non-blocking check: set a flag indicating whether previous sort tasks have finished.
                    bool previousSortTasksCompleted = true;
                    if (m_SortTasks != null)
                    {
                        int taskListSize = m_SortTasks.Length;
                        for (int i = 0; i < m_SortTasks.Length; i++)
                        {
                            var t = m_SortTasks[i];
                            if (t != null && !t.IsCompleted)
                            {
                                previousSortTasksCompleted = false;
                                break;
                            }
                        }
                    }
                    // Start the new parallel sort workers without blocking.
                    if(previousSortTasksCompleted)
                        m_SortTasks = ParallelSortVisibleNodes(camPosition);
                    // Sort node references by distance (near-to-far for front-to-back rendering) while parallel work happens
                    m_VisibleNodeRefs.Sort((a, b) => a.distance.CompareTo(b.distance)); // Front-to-back
                    // Now join the tasks after doing useful work on main thread (if desired)
                    // JoinParallelSortThreads(m_SortTasks);
                }
            }
            else
            {
                // Sequential path: sort nodes first, then process
                m_VisibleNodeRefs.Sort((a, b) => a.distance.CompareTo(b.distance)); // Front-to-back
                // Sequential path: sort outliers (background elements processed last in front-to-back)
                if (m_OthersIndices.Count > 0)
                {
                    SortOutliers(camPosition);
                }
                // Limit sorting to closest nodes per frame for better performance during camera movement
                int nodesToSort = Mathf.Min(m_VisibleNodeRefs.Count, maxSortNodesPerFrame);
                int nodesSorted = 0;
                for (int i = 0; i < m_VisibleNodeRefs.Count && nodesSorted < nodesToSort; i++) // Front-to-back processing
                {
                    var nodeRef = m_VisibleNodeRefs[i];
                    if (SortNodeSplats(nodeRef.nodeIndex, camPosition))
                    {
                        nodesSorted++;
                    }
                }
            }
            // Append nodes in distance order (their lists now internally sorted and persistent)
            int currentIndex = 0;
            
            // First, add node splats (front elements for front-to-back rendering)
            for (int i = 0; i < m_VisibleNodeRefs.Count; i++)
            {
                var nodeRef = m_VisibleNodeRefs[i];
                var node = m_Nodes[nodeRef.nodeIndex];
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    // Ensure we have enough space
                    if (currentIndex + node.splatIndices.Count > m_VisibleSplatIndices.Length)
                    {
                        if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
                        {
                            visibleSplatCount = currentIndex;
                            UpdateVisibleIndicesBuffer();
                            return;
                        }
                    }
                    
                    // Copy node splat indices
                    for (int j = 0; j < node.splatIndices.Count; j++)
                    {
                        m_VisibleSplatIndices[currentIndex + j] = node.splatIndices[j];
                    }
                    currentIndex += node.splatIndices.Count;
                }
            }
            
            // Finally, add outliers (background elements for front-to-back rendering)
            if (m_OthersIndices.Count > 0)
            {
                if (currentIndex + m_OthersIndices.Count > m_VisibleSplatIndices.Length)
                {
                    EnsureVisibleSplatIndicesCapacity(currentIndex + m_OthersIndices.Count);
                    if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
                    {
                        visibleSplatCount = currentIndex;
                        UpdateVisibleIndicesBuffer();
                        return;
                    }
                }
                
                for (int i = 0; i < m_OthersIndices.Count; i++)
                {
                    m_VisibleSplatIndices[currentIndex + i] = m_OthersIndices[i];
                }
                currentIndex += m_OthersIndices.Count;
            }
            
            visibleSplatCount = currentIndex;
            UpdateVisibleIndicesBuffer();
        }

        // Thread-safe per-node sorting using local scratch arrays (no shared m_DistanceSortArray)
        void SortSplatsInNodeThreadSafe(List<int> splatIndices, Vector3 camPosition)
        {
            int count = splatIndices.Count;
            if (count <= 1)
                return;
            var scratch = ArrayPool<(float distance, int index)>.Shared.Rent(Mathf.NextPowerOfTwo(count));
            try
            {
                for (int i = 0; i < count; i++)
                {
                    int originalSplatIdx = splatIndices[i];
                    if (TryGetSplatPosition(originalSplatIdx, out float3 splatPos))
                    {
                        float distance = ((Vector3)splatPos - camPosition).sqrMagnitude;
                        scratch[i] = (distance, originalSplatIdx);
                    }
                    else
                    {
                        scratch[i] = (0f, originalSplatIdx);
                    }
                }
                System.Array.Sort(scratch, 0, count, System.Collections.Generic.Comparer<(float distance, int index)>.Create((a, b) => a.distance.CompareTo(b.distance))); // Front-to-back
                for (int i = 0; i < count; i++)
                    splatIndices[i] = scratch[i].index;
            }
            finally
            {
                ArrayPool<(float distance, int index)>.Shared.Return(scratch);
            }
        }

        bool ShouldResortOutliers(Vector3 camPosition)
        {
            if (m_OthersIndices.Count == 0)
                return false;
            if (!m_OthersSorted)
                return true;
            float baseThreshold = Mathf.Max(minOutlierResortDistance, m_OutlierRingRadius * outlierResortMoveFraction);
            float sqMove = (camPosition - m_LastOthersSortCamPos).sqrMagnitude;
            return sqMove >= baseThreshold * baseThreshold;
        }

        void SortOutliers(Vector3 camPosition)
        {
            if (!ShouldResortOutliers(camPosition))
                return;
            if (m_OthersIndices.Count > 1)
                SortSplatsInNode(m_OthersIndices, camPosition);
            // Removed native buffer sync to reduce overhead
            m_OthersSorted = true;
            m_LastOthersSortCamPos = camPosition;
        }

        public void SetOutlierResortFraction(float fraction, float minDistance = 0.05f)
        {
            outlierResortMoveFraction = Mathf.Max(0f, fraction);
            minOutlierResortDistance = Mathf.Max(0f, minDistance);
        }

        Task[] ParallelSortVisibleNodes(Vector3 camPosition)
        {
            // Snapshot the visible node refs to avoid concurrent access from parallel tasks
            var snapshot = m_VisibleNodeRefs.ToArray();
            int nodeCount = snapshot.Length;

            // Pre-filter nodes that actually need sorting for better task utilization
            var nodesToSort = new List<int>(); // Store indices into the snapshot array
            for (int i = 0; i < nodeCount; i++)
            {
                var nodeRef = snapshot[i];
                var node = m_Nodes[nodeRef.nodeIndex];
                if (node.splatIndices != null && node.splatIndices.Count > 1)
                {
                    // Check if already sorted for this camera direction (using angular threshold)
                    bool needsSort = !node.isSorted;
                    if (!needsSort)
                    {
                        Vector3 nodeCenter = node.bounds.center;
                        Vector3 oldDirection = (nodeCenter - node.lastSortCameraPosition).normalized;
                        Vector3 newDirection = (nodeCenter - oldDirection * node.maxExtent - camPosition).normalized;
                        float cosineAngle = Vector3.Dot(oldDirection, newDirection);
                        needsSort = cosineAngle < sortDirectionThreshold;
                    }

                    if (needsSort)
                        nodesToSort.Add(i);
                }
            }

            int sortNodeCount = nodesToSort.Count;
            bool haveOutliers = m_OthersIndices.Count > 0;
            bool needOutlierResort = haveOutliers && ShouldResortOutliers(camPosition);

            // Always run outlier sorting on a background task when needed
            Task CreateOutlierTaskIfNeeded()
            {
                if (!needOutlierResort)
                    return null;

                return Task.Run(() =>
                {
                    try
                    {
                        SortSplatsInNodeThreadSafe(m_OthersIndices, camPosition);
                        m_OthersSorted = true;
                        m_LastOthersSortCamPos = camPosition;
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"Parallel outlier sorting exception in worker task: {ex}");
                        try
                        {
                            SortSplatsInNode(m_OthersIndices, camPosition);
                            m_OthersSorted = true;
                            m_LastOthersSortCamPos = camPosition;
                        }
                        catch (Exception fallbackEx)
                        {
                            Debug.LogError($"Fallback outlier sequential sorting also failed: {fallbackEx}");
                        }
                    }
                });
            }

            if (sortNodeCount == 0)
            {
                // No nodes need sorting; just spawn outlier task if needed
                var outlierTask = CreateOutlierTaskIfNeeded();
                if (outlierTask != null)
                    return new Task[] { outlierTask };
                return null;
            }

            // Clamp desired worker count based on actual work
            int workers = parallelSortThreads;
            workers = Mathf.Min(workers, sortNodeCount); // not more workers than nodes to sort
            if (workers <= 1)
            {
                // Run the filtered node sorting on a background task
                Task seqTask = null;
                if (sortNodeCount > 0)
                {
                    seqTask = Task.Run(() =>
                    {
                        // Process from front to back for front-to-back rendering (closer nodes processed first)
                        for (int i = sortNodeCount - 1; i >= 0; i--)
                        {
                            int nodeRefIndex = nodesToSort[i];
                            var nodeRef = snapshot[nodeRefIndex];
                            var node = m_Nodes[nodeRef.nodeIndex];
                            SortSplatsInNodeThreadSafe(node.splatIndices, camPosition);
                            node.isSorted = true;
                            node.lastSortCameraPosition = camPosition;
                        }
                    });
                }

                // Create outlier task if needed
                var outlierTask = CreateOutlierTaskIfNeeded();

                // Return tasks array containing the sequential worker and optionally the outlier task
                if (seqTask != null && outlierTask != null)
                    return new Task[] { seqTask, outlierTask };
                if (seqTask != null)
                    return new Task[] { seqTask };
                if (outlierTask != null)
                    return new Task[] { outlierTask };
                return null; // No tasks to wait on
            }

            // Work-stealing parallel sort implementation
            // Shared work queue with thread-safe access - process from front to back for front-to-back rendering
            int nextWorkIndex = sortNodeCount - 1; // Start from the end (closest nodes)
            object workLock = new object();

            // Get next work item (thread-safe) - process from front to back for front-to-back rendering
            int GetNextWorkIndex()
            {
                lock (workLock)
                {
                    if (nextWorkIndex >= 0)
                        return nextWorkIndex--;
                    return -1; // No more work
                }
            }

            // Determine up-front whether we need a dedicated outlier task so we can size the tasks array correctly
            bool outlierTaskNeeded = needOutlierResort;

            // Create worker tasks that pull work as needed
            Task[] tasks = new Task[workers + (outlierTaskNeeded ? 1 : 0)];
            
            for (int w = 0; w < workers; w++)
            {
                tasks[w] = Task.Run(() =>
                {
                    try
                    {
                        int workIndex;
                        while ((workIndex = GetNextWorkIndex()) != -1)
                        {
                            int nodeRefIndex = nodesToSort[workIndex];
                            var nodeRef = snapshot[nodeRefIndex];
                            var node = m_Nodes[nodeRef.nodeIndex];
                            SortSplatsInNodeThreadSafe(node.splatIndices, camPosition);
                            node.isSorted = true;
                            node.lastSortCameraPosition = camPosition;
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"Parallel splat sorting exception in worker task: {ex}");
                        // Continue processing remaining work items with fallback method
                        int workIndex;
                        while ((workIndex = GetNextWorkIndex()) != -1)
                        {
                            int nodeRefIndex = nodesToSort[workIndex];
                            var nodeRef = snapshot[nodeRefIndex];
                            var node = m_Nodes[nodeRef.nodeIndex];
                            try
                            {
                                SortSplatsInNode(node.splatIndices, camPosition);
                                node.isSorted = true;
                                node.lastSortCameraPosition = camPosition;
                            }
                            catch (Exception fallbackEx)
                            {
                                Debug.LogError($"Fallback sequential sorting also failed: {fallbackEx}");
                            }
                        }
                    }
                });
            }

            // Spawn outlier task if needed and place at the end
            if (outlierTaskNeeded)
            {
                var outlierTask = CreateOutlierTaskIfNeeded();
                tasks[workers] = outlierTask;
            }
            return tasks;
        }

        void JoinParallelSortThreads(Task[] tasks)
        {
            if (tasks == null) return;
            try
            {
                Task.WaitAll(tasks);
            }
            catch (AggregateException ex)
            {
                Debug.LogError($"One or more parallel sorting tasks threw exceptions: {ex}");
            }
        }

        /// <summary>
        /// Start native sorting jobs for WebGL platform.
        /// </summary>
        /// <returns>Number of jobs started</returns>
        int StartNativeSortJobs(Vector3 camPosition)
        {
            if (!NativeSorting.IsAvailable)
                return 0;

            int jobsStarted = 0;

            // Collect nodes that need sorting
            var nodesToSort = new List<int>();
            for (int i = 0; i < m_VisibleNodeRefs.Count; i++)
            {
                var nodeRef = m_VisibleNodeRefs[i];
                var node = m_Nodes[nodeRef.nodeIndex];
                if (node.splatIndices != null && node.splatIndices.Count > 1)
                {
                    // Check if already sorted for this camera direction
                    bool needsSort = !node.isSorted;
                    if (!needsSort)
                    {
                        Vector3 nodeCenter = node.bounds.center;
                        Vector3 oldDirection = (nodeCenter - node.lastSortCameraPosition).normalized;
                        Vector3 newDirection = (nodeCenter - oldDirection * node.maxExtent - camPosition).normalized;
                        float cosineAngle = Vector3.Dot(oldDirection, newDirection);
                        needsSort = cosineAngle < sortDirectionThreshold;
                    }

                    if (needsSort)
                    {
                        nodesToSort.Add(nodeRef.nodeIndex);
                    }
                }
            }

            // Start native sort jobs for outliers if needed
            if (ShouldResortOutliers(camPosition) && m_OthersIndices.Count > 1)
            {
                if (StartNativeOutlierSort(camPosition))
                    jobsStarted++;
            }

            // Start native jobs for nodes that need sorting
            for (int i = nodesToSort.Count - 1; i >= 0; i--)
            {
                int nodeIndex = nodesToSort[i];
                if (StartNativeNodeSort(nodeIndex, camPosition))
                    jobsStarted++;
            }
            
            return jobsStarted;
        }

        /// <summary>
        /// Start a native sort job for outlier splats.
        /// </summary>
        /// <returns>True if job was started successfully</returns>
        bool StartNativeOutlierSort(Vector3 camPosition)
        {
            if (m_OthersIndices.Count <= 1) return false;
            if (!m_AllPositionsNativeValid) return false;
            try
            {
                bool usedPersistentInput = EnsureOutlierNativeIndices();
                NativeArray<int> input;
                if (usedPersistentInput)
                {
                    input = m_OthersNativeIndices;
                }
                else
                {
                    input = new NativeArray<int>(m_OthersIndices.Count, Allocator.Persistent);
                    for (int i = 0; i < m_OthersIndices.Count; i++) input[i] = m_OthersIndices[i];
                }
                var output = new NativeArray<int>(m_OthersIndices.Count, Allocator.Persistent);
                var handle = NativeSorting.StartSortJob(input, m_AllPositionsNative, output, camPosition);
                if (!handle.IsValid)
                {
                    if (!usedPersistentInput && input.IsCreated) input.Dispose();
                    if (output.IsCreated) output.Dispose();
                    return false;
                }
                m_NativeSortJobs.Add(handle);
                m_NativeJobInfos.Add(new NativeSortJobInfo
                {
                    isOutlierJob = true,
                    nodeIndex = -1,
                    cameraPosition = camPosition,
                    inputIndices = input,
                    sortedIndices = output,
                    disposeInput = !usedPersistentInput
                });
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to start native outlier sort job: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Start a native sort job for a specific node.
        /// </summary>
        /// <returns>True if job was started successfully</returns>
        bool StartNativeNodeSort(int nodeIndex, Vector3 camPosition)
        {
            if (nodeIndex < 0 || nodeIndex >= m_Nodes.Count) return false;
            var node = m_Nodes[nodeIndex];
            if (node.splatIndices == null || node.splatIndices.Count <= 1) return false;
            if (!m_AllPositionsNativeValid) return false;
            try
            {
                // Ensure persistent native splat index buffer exists for this node (lazy init)
                bool usedPersistentInput = EnsureNodeNativeIndices(nodeIndex);
                NativeArray<int> input;
                if (usedPersistentInput)
                {
                    input = node.nativeSplatIndices; // already valid & persistent
                }
                else
                {
                    // Fallback (should rarely happen) allocate one-shot buffer
                    input = new NativeArray<int>(node.splatIndices.Count, Allocator.Persistent);
                    for (int i = 0; i < node.splatIndices.Count; i++) input[i] = node.splatIndices[i];
                }
                var output = new NativeArray<int>(node.splatIndices.Count, Allocator.Persistent);
                var handle = NativeSorting.StartSortJob(input, m_AllPositionsNative, output, camPosition);
                if (!handle.IsValid)
                {
                    if (!usedPersistentInput && input.IsCreated) input.Dispose();
                    if (output.IsCreated) output.Dispose();
                    return false;
                }
                m_NativeSortJobs.Add(handle);
                m_NativeJobInfos.Add(new NativeSortJobInfo
                {
                    isOutlierJob = false,
                    nodeIndex = nodeIndex,
                    cameraPosition = camPosition,
                    inputIndices = input,
                    sortedIndices = output,
                    disposeInput = !usedPersistentInput
                });
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to start native sort job for node {nodeIndex}: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Collect results from completed native sort jobs.
        /// </summary>
        /// <returns>True if all native jobs are completed, false if any are still running</returns>
        bool CollectNativeSortResults()
        {
            for (int i = m_NativeSortJobs.Count - 1; i >= 0; i--)
            {
                var handle = m_NativeSortJobs[i];
                if (!handle.IsCompleted) continue;
                var info = m_NativeJobInfos[i];
                try
                {
                    if (info.isOutlierJob)
                        ApplySortedOutlierResults(info.sortedIndices, info.cameraPosition);
                    else
                        ApplySortedNodeResults(info.nodeIndex, info.sortedIndices, info.cameraPosition);
                    try { NativeSorting.CleanupJob(handle); } catch { }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error collecting native sort results: {ex.Message}");
                }
                try { if (info.disposeInput && info.inputIndices.IsCreated) info.inputIndices.Dispose(); } catch { }
                try { if (info.sortedIndices.IsCreated) info.sortedIndices.Dispose(); } catch { }
                m_NativeSortJobs.RemoveAt(i);
                m_NativeJobInfos.RemoveAt(i);
            }
            return m_NativeSortJobs.Count == 0;
        }
        
        /// <summary>
        /// Apply sorted results to outlier indices.
        /// </summary>
        void ApplySortedOutlierResults(NativeArray<int> sortedIndices, Vector3 camPosition)
        {
            m_OthersIndices.Clear();
            for (int i = 0; i < sortedIndices.Length; i++)
                m_OthersIndices.Add(sortedIndices[i]);
            // Removed optional native buffer refresh to avoid overhead
            m_OthersSorted = true;
            m_LastOthersSortCamPos = camPosition;
        }
        
        /// <summary>
        /// Apply sorted results to node indices.
        /// </summary>
        void ApplySortedNodeResults(int nodeIndex, NativeArray<int> sortedIndices, Vector3 camPosition)
        {
            if (nodeIndex < 0 || nodeIndex >= m_Nodes.Count)
            {
                Debug.LogWarning($"Invalid node index for native sort results: {nodeIndex}");
                return;
            }
            var node = m_Nodes[nodeIndex];
            if (node.splatIndices == null)
            {
                Debug.LogWarning($"Node {nodeIndex} has null splat indices");
                return;
            }
            node.splatIndices.Clear();
            for (int i = 0; i < sortedIndices.Length; i++)
                node.splatIndices.Add(sortedIndices[i]);
            // Removed optional native indices refresh
            node.isSorted = true;
            node.lastSortCameraPosition = camPosition;
        }

        /// <summary>
        /// Cleanup completed native jobs without applying results.
        /// </summary>
        void CleanupCompletedNativeJobs()
        {
            for (int i = m_NativeSortJobs.Count - 1; i >= 0; i--)
            {
                var handle = m_NativeSortJobs[i];
                if (!handle.IsCompleted) continue;
                try { NativeSorting.CleanupJob(handle); } catch { }
                var info = m_NativeJobInfos[i];
                try { if (info.disposeInput && info.inputIndices.IsCreated) info.inputIndices.Dispose(); } catch { }
                try { if (info.sortedIndices.IsCreated) info.sortedIndices.Dispose(); } catch { }
                m_NativeSortJobs.RemoveAt(i);
                m_NativeJobInfos.RemoveAt(i);
            }
        }

        /// <summary>
        /// Sort splats in a node and mark it as sorted for the current camera view.
        /// </summary>
        public bool SortNodeSplats(int nodeIndex, Vector3 camPosition, bool forceSort = false)
        {
            if (nodeIndex < 0 || nodeIndex >= m_Nodes.Count) return false;
            var node = m_Nodes[nodeIndex];
            if (node.splatIndices == null || node.splatIndices.Count <= 1) return false;

            if (!forceSort && node.isSorted)
            {
                Vector3 nodeCenter = node.bounds.center;
                Vector3 oldDirection = (nodeCenter - node.lastSortCameraPosition).normalized;
                Vector3 newDirection = (nodeCenter - oldDirection * node.maxExtent - camPosition).normalized;

                float cosineAngle = Vector3.Dot(oldDirection, newDirection);
                if (cosineAngle >= sortDirectionThreshold)
                    return false;
            }

            SortSplatsInNode(node.splatIndices, camPosition);
            node.isSorted = true;
            node.lastSortCameraPosition = camPosition;
            return true;
        }

        /// <summary>
        /// Mark all nodes and outliers as needing re-sort.
        /// </summary>
        public void InvalidateAllSorts()
        {
            for (int i = 0; i < m_Nodes.Count; i++)
            {
                m_Nodes[i].isSorted = false;
            }
            m_OthersSorted = false;
        }

        /// <summary>
        /// Set the sort direction threshold using angle in degrees for easier configuration.
        /// </summary>
        public void SetSortDirectionThresholdDegrees(float angleDegrees)
        {
            sortDirectionThreshold = Mathf.Cos(angleDegrees * Mathf.Deg2Rad);
        }

        void SortSplatsInNode(List<int> splatIndices, Vector3 camPosition)
        {
            int count = splatIndices.Count;
            if (count <= 1) return;
            if (m_DistanceSortArray == null || m_DistanceSortArray.Length < count)
                m_DistanceSortArray = new (float distance, int index)[Mathf.NextPowerOfTwo(count)];
            for (int i = 0; i < count; i++)
            {
                int originalSplatIdx = splatIndices[i];
                if (TryGetSplatPosition(originalSplatIdx, out float3 splatPos))
                {
                    float distance = ((Vector3)splatPos - camPosition).sqrMagnitude;
                    m_DistanceSortArray[i] = (distance, originalSplatIdx);
                }
                else
                {
                    m_DistanceSortArray[i] = (0f, originalSplatIdx);
                }
            }
            System.Array.Sort(m_DistanceSortArray, 0, count, System.Collections.Generic.Comparer<(float distance, int index)>.Create((a, b) => a.distance.CompareTo(b.distance))); // Front-to-back
            for (int i = 0; i < count; i++)
                splatIndices[i] = m_DistanceSortArray[i].index;
        }

        void CollectVisibleNodesWithDistance(int nodeIndex, Plane[] frustumPlanes, Vector3 camPosition)
        {
            m_TraversalStack.Clear();
            
            // Early exit if invalid starting node
            if (nodeIndex >= m_Nodes.Count)
                return;
                
            m_TraversalStack.Push(nodeIndex);
            
            while (m_TraversalStack.Count > 0)
            {
                int currentNodeIndex = m_TraversalStack.Pop();
                
                // Bounds check
                if (currentNodeIndex >= m_Nodes.Count)
                    continue;
                    
                var node = m_Nodes[currentNodeIndex];
                
                // Frustum culling - early exit if node not visible
                if (!GeometryUtility.TestPlanesAABB(frustumPlanes, node.bounds))
                    continue;
                
                if (node.isLeaf)
                {
                    // Add leaf node if it has splats
                    if (node.splatIndices != null && node.splatIndices.Count > 0)
                    {
                        float nodeDistance = (node.center - camPosition).sqrMagnitude;
                        m_VisibleNodeRefs.Add(new VisibleNodeRef
                        {
                            distance = nodeDistance,
                            nodeIndex = currentNodeIndex
                        });
                    }
                }
                else if (node.childIndices != null)
                {
                    // Add children to stack for traversal (reverse order for consistent traversal)
                    for (int i = node.childIndices.Count - 1; i >= 0; i--)
                    {
                        int childIndex = node.childIndices[i];
                        if (childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];
                            // Only traverse children that have content or are internal nodes
                            if ((childNode.splatIndices != null && childNode.splatIndices.Count > 0) || !childNode.isLeaf)
                            {
                                m_TraversalStack.Push(childIndex);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Sequential fallback for nodes/outliers not handled by native jobs.
        /// </summary>
        void SequentialSortFallback(Vector3 camPosition)
        {
            // Check if outliers need sequential sorting (if not already handled by native job)
            if (ShouldResortOutliers(camPosition))
            {
                SortOutliers(camPosition);
            }
            
            // Check nodes that may not have been processed by native jobs
            for (int i = 0; i < m_VisibleNodeRefs.Count; i++)
            {
                var nodeRef = m_VisibleNodeRefs[i];
                var node = m_Nodes[nodeRef.nodeIndex];
                
                // Only process nodes that aren't already sorted
                if (!node.isSorted && node.splatIndices != null && node.splatIndices.Count > 1)
                {
                    SortNodeSplats(nodeRef.nodeIndex, camPosition);
                }
            }
        }

        bool EnsureNodeNativeIndices(int nodeIndex)
        {
            if (nodeIndex < 0 || nodeIndex >= m_Nodes.Count) return false;
            var node = m_Nodes[nodeIndex];
            if (node.splatIndices == null || node.splatIndices.Count == 0) return false;
            if (!node.nativeIndicesValid || !node.nativeSplatIndices.IsCreated || node.nativeSplatIndices.Length != node.splatIndices.Count)
            {
                // Dispose previous if size mismatch
                if (node.nativeIndicesValid && node.nativeSplatIndices.IsCreated)
                {
                    try { node.nativeSplatIndices.Dispose(); } catch {}
                }
                try
                {
                    node.nativeSplatIndices = new NativeArray<int>(node.splatIndices.Count, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    for (int i = 0; i < node.splatIndices.Count; i++)
                        node.nativeSplatIndices[i] = node.splatIndices[i];
                    node.nativeIndicesValid = true;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"Failed to allocate native splat index buffer for node {nodeIndex}: {ex.Message}");
                    node.nativeIndicesValid = false;
                }
            }
            return node.nativeIndicesValid;
        }

        bool EnsureOutlierNativeIndices()
        {
            if (m_OthersIndices.Count == 0) return false;
            if (!m_OthersNativeValid || !m_OthersNativeIndices.IsCreated || m_OthersNativeIndices.Length != m_OthersIndices.Count)
            {
                if (m_OthersNativeValid && m_OthersNativeIndices.IsCreated)
                {
                    try { m_OthersNativeIndices.Dispose(); } catch {}
                }
                try
                {
                    m_OthersNativeIndices = new NativeArray<int>(m_OthersIndices.Count, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    for (int i = 0; i < m_OthersIndices.Count; i++)
                        m_OthersNativeIndices[i] = m_OthersIndices[i];
                    m_OthersNativeValid = true;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"Failed to allocate native outlier index buffer: {ex.Message}");
                    m_OthersNativeValid = false;
                }
            }
            return m_OthersNativeValid;
        }
    }
}
