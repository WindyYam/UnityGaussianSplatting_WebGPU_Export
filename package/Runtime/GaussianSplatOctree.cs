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
            // For internal nodes this may be null or empty.
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
        }

        public struct SplatInfo
        {
            public float3 position;
            public int originalIndex;
        }

        readonly List<OctreeNode> m_Nodes = new();
        readonly List<SplatInfo> m_SplatInfos = new();
        // Note: per-node splat indices are stored inside OctreeNode.splatIndices for leaf nodes.
        readonly List<int> m_VisibleSplatIndices = new();

        // Dictionary to map original splat indices to their positions for efficient lookup during sorting
        readonly Dictionary<int, float3> m_OriginalIndexToPosition = new();

        // Configuration
        int m_MaxDepth;
        int m_MaxSplatsPerLeaf;
        Bounds m_RootBounds;
        bool m_Built;

        // GPU buffer for visible splat indices (updated per frame/N frames)
        GraphicsBuffer m_VisibleIndicesBuffer;

        // Outlier splat indices that lie outside the main root bounds (always included in culling)
        readonly List<int> m_OthersIndices = new();

        // Reusable array for distance sorting to avoid allocations
        (float distance, int index)[] m_DistanceSortArray;

        // Structure to store visible node references with their distance for hierarchical sorting
        struct VisibleNodeRef
        {
            public float distance;
            public int nodeIndex; // Index into m_Nodes instead of copying splat indices
        }

        // Reusable list for visible node references during sorting
        readonly List<VisibleNodeRef> m_VisibleNodeRefs = new();        // Enable / disable parallel sorting (public for runtime tuning)
        public bool enableParallelSorting = true;
        // Configurable number of worker threads for node sorting (excluding main thread)
        public int parallelSortThreads = 8; // Default sort threads, Safe for most of the platform
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

        public int nodeCount => m_Nodes.Count;
        public int totalSplats => m_SplatInfos.Count;
        public bool isBuilt => m_Built;
        public GraphicsBuffer visibleIndicesBuffer => m_VisibleIndicesBuffer;
        public int visibleSplatCount { get; private set; }

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

            // In WebGPU platform, SystemInfo.processorCount is not reliable.
            bool isWebPlatform = Application.platform == RuntimePlatform.WebGLPlayer;
            if (isWebPlatform)
            {
                // WebGL / Web platform does not reliably support multithreading in Unity.
                // Disable parallel sorting and fall back to single-threaded path.
                enableParallelSorting = false;
                Debug.LogWarning("GaussianSplatOctree: Web platform detected — threading is not supported in Unity(unfortunately). Disabling parallel sorting and using single-threaded sorting fallback.");
            }
            else
            {
                int reportedCores = SystemInfo.processorCount;
                if (reportedCores > 0)
                    parallelSortThreads = reportedCores;
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
            float3 com = float3.zero;
            for (int i = 0; i < total; i++)
                com += splatPositions[i];
            com /= total;

            var distList = new List<(int idx, float d)>();
            distList.Capacity = total;
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

            m_SplatInfos.Clear();
            m_SplatInfos.Capacity = total;

            // Build lookup dictionary for original indices to positions
            m_OriginalIndexToPosition.Clear();
            for (int i = 0; i < total; i++)
            {
                int src = distList[i].idx;
                var position = splatPositions[src];
                m_SplatInfos.Add(new SplatInfo { position = position, originalIndex = src });
                m_OriginalIndexToPosition[src] = position;
            }

            // Create root bounds based on the inCount splats (centered on center-of-mass)
            Bounds rootBounds;
            if (inCount > 0)
            {
                float3 min = m_SplatInfos[0].position;
                float3 max = m_SplatInfos[0].position;
                for (int i = 1; i < inCount; i++)
                {
                    min = math.min(min, m_SplatInfos[i].position);
                    max = math.max(max, m_SplatInfos[i].position);
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
            for (int i = 0; i < inCount; i++) rootSplatList.Add(i); // indices into m_SplatInfos
            BuildRecursive(0, 0, rootSplatList);

            // Handle remaining outliers: put their original indices into m_SplatIndices and track them in m_OthersIndices
            m_OthersIndices.Clear();
            if (othersCount > 0)
            {
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = m_SplatInfos[inCount + i].originalIndex;
                    m_OthersIndices.Add(orig);
                }
            }
            m_OthersSorted = false; // reset outlier sorting state after build
            m_LastOthersSortCamPos = Vector3.zero;
            // Compute average outlier ring radius (ignore min/max & extra stats for simplicity)
            m_OutlierRingRadius = 0f;
            if (othersCount > 0)
            {
                Vector3 center = m_RootBounds.center;
                double accum = 0.0;
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = m_SplatInfos[inCount + i].originalIndex;
                    if (m_OriginalIndexToPosition.TryGetValue(orig, out float3 p))
                    {
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
        }

        void BuildRecursive(int nodeIndex, int depth, List<int> splatList)
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
                    if (infoIdx < 0 || infoIdx >= m_SplatInfos.Count)
                    {
                        Debug.LogError($"Octree leaf node splat info index out of bounds: {infoIdx} >= {m_SplatInfos.Count}");
                        continue;
                    }
                    node.splatIndices.Add(m_SplatInfos[infoIdx].originalIndex);
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
                if (infoIdx < 0 || infoIdx >= m_SplatInfos.Count)
                {
                    Debug.LogError($"Octree splat distribution info index out of bounds: {infoIdx} >= {m_SplatInfos.Count}");
                    continue;
                }

                var splat = m_SplatInfos[infoIdx];

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
                    BuildRecursive(childNodeIndex, depth + 1, childSplatsIdx[i]);
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
                // For leaf nodes, calculate tight bounds based on actual splat positions
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    // Get first splat position to initialize bounds
                    int firstSplatIdx = node.splatIndices[0];
                    if (m_OriginalIndexToPosition.TryGetValue(firstSplatIdx, out float3 firstPos))
                    {
                        float3 min = firstPos;
                        float3 max = firstPos;

                        // Expand bounds to include all splats in this leaf
                        for (int i = 1; i < node.splatIndices.Count; i++)
                        {
                            int splatIdx = node.splatIndices[i];
                            if (m_OriginalIndexToPosition.TryGetValue(splatIdx, out float3 pos))
                            {
                                min = math.min(min, pos);
                                max = math.max(max, pos);
                            }
                        }

                        // Update node bounds with tight fit
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
                return false; // No splats, bounds unchanged
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

            m_VisibleSplatIndices.Clear();

            // Extract frustum planes from camera
            var frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);

            // Traverse octree and collect visible splats
            CullNodeRecursive(0, frustumPlanes);

            // Always include 'others' outlier splats
            if (m_OthersIndices.Count > 0)
            {
                // Fast bulk append outlier indices
                m_VisibleSplatIndices.AddRange(m_OthersIndices);
            }

            visibleSplatCount = m_VisibleSplatIndices.Count;

            // Debug log culling performance
            //float cullingRatio = (float)visibleSplatCount / Mathf.Max(1, m_SplatInfos.Count);
            //Debug.Log($"Octree culling: {visibleSplatCount}/{m_SplatInfos.Count} splats visible ({cullingRatio:P1})");

            // Update GPU buffer
            UpdateVisibleIndicesBuffer();

            return visibleSplatCount;
        }

        void CullNodeRecursive(int nodeIndex, Plane[] frustumPlanes)
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
                    m_VisibleSplatIndices.AddRange(node.splatIndices);
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
                                CullNodeRecursive(childIndex, frustumPlanes);
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

            // Convert to uint array and upload visible indices
            var nativeArray = new NativeArray<uint>(visibleSplatCount, Allocator.Temp);
            for (int i = 0; i < visibleSplatCount; i++)
            {
                nativeArray[i] = (uint)m_VisibleSplatIndices[i];
            }
            m_VisibleIndicesBuffer.SetData(nativeArray, 0, 0, visibleSplatCount);
            nativeArray.Dispose();
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
            m_Nodes.Clear();
            m_SplatInfos.Clear();
            m_VisibleSplatIndices.Clear();
            m_VisibleNodeRefs.Clear();
            m_OriginalIndexToPosition.Clear();
            m_VisibleIndicesBuffer?.Dispose();
            m_VisibleIndicesBuffer = null;
            m_DistanceSortArray = null; // Release sort array memory
            visibleSplatCount = 0;
            m_Built = false;
            m_OthersIndices.Clear();
            m_OthersSorted = false;
            m_LastOthersSortCamPos = Vector3.zero;
            m_OutlierRingRadius = 0f;
        }

        public void Dispose()
        {
            Clear();
        }

        /// <summary>
        /// Sort visible splat indices by 3D distance from camera (back-to-front for alpha blending).
        /// Hierarchical sorting optimization.
        /// </summary>
        public void SortVisibleSplatsByDepth(Camera camera)
        {
            if (!m_Built)
                return;
            var camPosition = camera.transform.position;
            m_VisibleSplatIndices.Clear();
            m_VisibleNodeRefs.Clear();
            var frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);
            CollectVisibleNodesWithDistance(0, frustumPlanes, camPosition);

            if (enableParallelSorting)
            {
                // Non-blocking check: set a flag indicating whether previous sort tasks have finished.
                // Do not block/wait here — caller can poll this flag if needed.
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
                // Sort node references by distance (far-to-near for back-to-front rendering) while parallel work happens
                m_VisibleNodeRefs.Sort((a, b) => b.distance.CompareTo(a.distance));
                // Now join the tasks after doing useful work on main thread (if desired)
                // JoinParallelSortThreads(m_SortTasks);
            }
            else
            {
                // Sequential path: sort nodes first, then process
                m_VisibleNodeRefs.Sort((a, b) => b.distance.CompareTo(a.distance));
                // Sequential path: outliers first (they are assumed farthest)
                if (m_OthersIndices.Count > 0)
                {
                    SortOutliers(camPosition);
                }
                for (int i = 0; i < m_VisibleNodeRefs.Count; i++)
                {
                    var nodeRef = m_VisibleNodeRefs[i];
                    SortNodeSplats(nodeRef.nodeIndex, camPosition);
                }
            }
            // Append nodes in distance order (their lists now internally sorted and persistent)
            m_VisibleSplatIndices.AddRange(m_OthersIndices);
            for (int i = 0; i < m_VisibleNodeRefs.Count; i++)
            {
                var nodeRef = m_VisibleNodeRefs[i];
                var node = m_Nodes[nodeRef.nodeIndex];
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                    m_VisibleSplatIndices.AddRange(node.splatIndices);
            }
            visibleSplatCount = m_VisibleSplatIndices.Count;
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
                    if (m_OriginalIndexToPosition.TryGetValue(originalSplatIdx, out float3 splatPos))
                    {
                        float distance = ((Vector3)splatPos - camPosition).sqrMagnitude;
                        scratch[i] = (distance, originalSplatIdx);
                    }
                    else
                    {
                        scratch[i] = (0f, originalSplatIdx);
                    }
                }
                System.Array.Sort(scratch, 0, count, System.Collections.Generic.Comparer<(float distance, int index)>.Create((a, b) => b.distance.CompareTo(a.distance)));
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
                        // Process from back to front for visual priority (closer nodes processed first)
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
            // Shared work queue with thread-safe access - process from back to front for visual priority
            int nextWorkIndex = sortNodeCount - 1; // Start from the end (closest nodes)
            object workLock = new object();

            // Get next work item (thread-safe) - process from back to front for visual priority
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
        /// Sort splats in a node and mark it as sorted for the current camera view.
        /// </summary>
        public void SortNodeSplats(int nodeIndex, Vector3 camPosition, bool forceSort = false)
        {
            if (nodeIndex < 0 || nodeIndex >= m_Nodes.Count) return;
            var node = m_Nodes[nodeIndex];
            if (node.splatIndices == null || node.splatIndices.Count <= 1) return;

            if (!forceSort && node.isSorted)
            {
                Vector3 nodeCenter = node.bounds.center;
                Vector3 oldDirection = (nodeCenter - node.lastSortCameraPosition).normalized;
                Vector3 newDirection = (nodeCenter - oldDirection * node.maxExtent - camPosition).normalized;

                float cosineAngle = Vector3.Dot(oldDirection, newDirection);
                if (cosineAngle >= sortDirectionThreshold)
                    return;
            }

            SortSplatsInNode(node.splatIndices, camPosition);
            node.isSorted = true;
            node.lastSortCameraPosition = camPosition;
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
                if (m_OriginalIndexToPosition.TryGetValue(originalSplatIdx, out float3 splatPos))
                {
                    float distance = ((Vector3)splatPos - camPosition).sqrMagnitude;
                    m_DistanceSortArray[i] = (distance, originalSplatIdx);
                }
                else
                {
                    m_DistanceSortArray[i] = (0f, originalSplatIdx);
                }
            }
            System.Array.Sort(m_DistanceSortArray, 0, count, System.Collections.Generic.Comparer<(float distance, int index)>.Create((a, b) => b.distance.CompareTo(a.distance)));
            for (int i = 0; i < count; i++)
                splatIndices[i] = m_DistanceSortArray[i].index;
        }

        void CollectVisibleNodesWithDistance(int nodeIndex, Plane[] frustumPlanes, Vector3 camPosition)
        {
            if (nodeIndex >= m_Nodes.Count)
                return;
            var node = m_Nodes[nodeIndex];
            if (!GeometryUtility.TestPlanesAABB(frustumPlanes, node.bounds))
                return;
            if (node.isLeaf)
            {
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    float nodeDistance = (node.center - camPosition).sqrMagnitude;
                    m_VisibleNodeRefs.Add(new VisibleNodeRef
                    {
                        distance = nodeDistance,
                        nodeIndex = nodeIndex
                    });
                }
            }
            else if (node.childIndices != null)
            {
                foreach (var childIndex in node.childIndices)
                {
                    if (childIndex < m_Nodes.Count)
                    {
                        var childNode = m_Nodes[childIndex];
                        if ((childNode.splatIndices != null && childNode.splatIndices.Count > 0) || !childNode.isLeaf)
                            CollectVisibleNodesWithDistance(childIndex, frustumPlanes, camPosition);
                    }
                }
            }
        }
    }
}
