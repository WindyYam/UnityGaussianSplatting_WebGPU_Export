// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.XR;
using Object = UnityEngine.Object;

namespace GaussianSplatting.Runtime
{
    class GaussianSplatRenderSystem
    {
        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker s_ProfDraw = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCompose = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfSort = new(ProfilerCategory.Render, "GaussianSplat.Sort", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static GaussianSplatRenderSystem instance => ms_Instance ??= new GaussianSplatRenderSystem();
        static GaussianSplatRenderSystem ms_Instance;

        readonly Dictionary<GaussianSplatRenderer, MaterialPropertyBlock> m_Splats = new();
        readonly HashSet<Camera> m_CameraCommandBuffersDone = new();
        readonly List<(GaussianSplatRenderer, MaterialPropertyBlock)> m_ActiveSplats = new();

        CommandBuffer m_CommandBuffer;
        GraphicsBuffer m_CubeIndexBuffer;
        GraphicsBuffer m_GlobalUniforms;
        Material m_MatSplats;
        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal Material m_MatComposite;
        Material m_MatDebugPoints;
        Material m_MatDebugBoxes;
        uint m_FrameOffset;
        GaussianSplatTemporalFilter m_TemporalFilter;

        // Public accessor to get or create the temporal filter instance.
        // Other renderer features (URP/HDRP) should call this to obtain the filter
        // instead of accessing internal fields directly.
        public GaussianSplatTemporalFilter GetTemporalFilter()
        {
            return m_TemporalFilter ??= new GaussianSplatTemporalFilter();
        }

        struct SplatGlobalUniforms // match cbuffer SplatGlobalUniforms in shaders
        {
            public uint transparencyMode;
            public uint frameOffset;
            public uint needMotionVectors;
            public uint padding0; // padding to make struct 16 bytes to match GPU cbuffer layout
        }

        public void RegisterSplat(GaussianSplatRenderer r)
        {
            if (m_Splats.Count == 0)
            {
                if (GraphicsSettings.currentRenderPipeline == null)
                    Camera.onPreCull += OnPreCullCamera;
            }

            var mpb = new MaterialPropertyBlock();
            // Set immutable splat data buffers once during registration if resources are ready
            r.SetAssetDataOnMaterial(mpb);

            m_Splats.Add(r, mpb);
        }

        public void UnregisterSplat(GaussianSplatRenderer r)
        {
            if (!m_Splats.ContainsKey(r))
                return;
            m_Splats.Remove(r);
            if (m_Splats.Count == 0)
                CleanupAfterAllSplatsDeleted();
        }

        public void UpdateSplatAssetData(GaussianSplatRenderer r)
        {
            if (m_Splats.TryGetValue(r, out var mpb))
            {
                r.SetAssetDataOnMaterial(mpb);
            }
        }

        void CleanupAfterAllSplatsDeleted()
        {
            if (m_CameraCommandBuffersDone != null)
            {
                if (m_CommandBuffer != null)
                {
                    foreach (var cam in m_CameraCommandBuffersDone)
                    {
                        if (cam)
                            cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                    }
                }
                m_CameraCommandBuffersDone.Clear();
            }

            m_ActiveSplats.Clear();
            m_CubeIndexBuffer?.Dispose();
            m_CubeIndexBuffer = null;
            m_CommandBuffer?.Dispose();
            m_CommandBuffer = null;
            m_GlobalUniforms?.Dispose();
            m_GlobalUniforms = null;
            Object.DestroyImmediate(m_MatSplats);
            Object.DestroyImmediate(m_MatComposite);
            Object.DestroyImmediate(m_MatDebugPoints);
            Object.DestroyImmediate(m_MatDebugBoxes);
            m_TemporalFilter?.Dispose();
            m_TemporalFilter = null;
            // Cleanup static dummy buffers when no more splats exist
            GaussianSplatRenderer.DisposeDummyBuffers();
            Camera.onPreCull -= OnPreCullCamera;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            m_ActiveSplats.Clear();
            foreach (var kvp in m_Splats)
            {
                var gs = kvp.Key;
                if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidAsset || !gs.HasValidRenderSetup)
                    continue;
                m_ActiveSplats.Add((kvp.Key, kvp.Value));
            }
            if (m_ActiveSplats.Count == 0)
                return false;

            // sort them by order and depth from camera
            var camTr = cam.transform;
            m_ActiveSplats.Sort((a, b) =>
            {
                var orderA = a.Item1.m_RenderOrder;
                var orderB = b.Item1.m_RenderOrder;
                if (orderA != orderB)
                    return orderB.CompareTo(orderA);
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public void SortAllSplats(Camera cam, CommandBuffer cmb)
        {
            if (cam.cameraType == CameraType.Preview)
                return;
            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            if (!settings.needSorting)
                return; // no need to sort

            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;
                var matrix = gs.transform.localToWorldMatrix;
                
                // Increment frame counter for all modes
                ++gs.m_FrameCounter;
                
                // Note: For alpha blend mode with octree culling, sorting is now done 
                // in PerformOctreeCulling() as part of the hierarchical optimization
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public void RenderAllSplats(Camera cam, CommandBuffer cmb)
        {
            EnsureMaterials();
            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            Material displayMat = settings.m_RenderMode switch
            {
                DebugRenderMode.DebugPoints => m_MatDebugPoints,
                DebugRenderMode.DebugPointIndices => m_MatDebugPoints,
                DebugRenderMode.DebugBoxes => m_MatDebugBoxes,
                DebugRenderMode.DebugChunkBounds => m_MatDebugBoxes,
                _ => m_MatSplats
            };
            if (displayMat == null)
                return;

            EnsureCubeIndexBuffer();

            m_GlobalUniforms ??= new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, UnsafeUtility.SizeOf<SplatGlobalUniforms>());
            NativeArray<SplatGlobalUniforms> sgu = new(1, Allocator.Temp);
            sgu[0] = new SplatGlobalUniforms { transparencyMode = (uint)settings.m_Transparency, frameOffset = m_FrameOffset, needMotionVectors = (uint)settings.m_TemporalFilter};
            cmb.SetBufferData(m_GlobalUniforms, sgu);
            m_FrameOffset++;

            // Set blend mode based on transparency mode
            if (settings.isDebugRender)
            {
                // Debug rendering uses different blend settings
                displayMat.SetInt(GaussianSplatRenderer.Props.SrcBlend, (int)BlendMode.OneMinusDstAlpha);
                displayMat.SetInt(GaussianSplatRenderer.Props.DstBlend, (int)BlendMode.One);
                displayMat.SetInt(GaussianSplatRenderer.Props.ZWrite, 0);
            }
            else
            {
                // Normal splat rendering - choose blend mode based on transparency mode
                switch (settings.m_Transparency)
                {
                    case TransparencyMode.Stochastic:
                    case TransparencyMode.StochasticHalfTone:
                        // Stochastic transparency with binary alpha
                        displayMat.SetInt(GaussianSplatRenderer.Props.SrcBlend, (int)BlendMode.One);
                        displayMat.SetInt(GaussianSplatRenderer.Props.DstBlend, (int)BlendMode.Zero);
                        displayMat.SetInt(GaussianSplatRenderer.Props.ZWrite, 1);
                        break;
                    
                    case TransparencyMode.AlphaBlend:
                        // Premultiplied alpha blending (shader does: i.col.rgb *= alpha)
                        displayMat.SetInt(GaussianSplatRenderer.Props.SrcBlend, (int)BlendMode.One);
                        displayMat.SetInt(GaussianSplatRenderer.Props.DstBlend, (int)BlendMode.OneMinusSrcAlpha);
                        displayMat.SetInt(GaussianSplatRenderer.Props.ZWrite, 0);
                        break;
                }
            }

            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;

                var matrix = gs.transform.localToWorldMatrix;

                var mpb = kvp.Item2;
                // No need to clear and reset immutable buffers - they were set during registration
                // Only set per-frame varying properties

                Matrix4x4 matView = cam.worldToCameraMatrix;
                Matrix4x4 matO2W = matrix;
                Matrix4x4 matW2O = matrix.inverse;
                Matrix4x4 currentMatMV = matView * matO2W;
                int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
                int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
                Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
                Vector4 camPos = cam.transform.position;
                
                mpb.SetMatrix(GaussianSplatRenderer.Props.MatrixMV, currentMatMV);
                mpb.SetMatrix(GaussianSplatRenderer.Props.PrevMatrixMV, gs.m_PrevMatrixMV);
                // Compute approximate previous view matrix. Splat objects are static,
                // so we assume object-to-world hasn't changed. Approximate prevView = prevMV * inverse(currentObjectToWorld) = prevMV * matW2O.
                Matrix4x4 prevView = gs.m_PrevMatrixMV * matW2O;
                mpb.SetMatrix(GaussianSplatRenderer.Props.PrevMatrixV, prevView);
                mpb.SetMatrix(GaussianSplatRenderer.Props.MatrixObjectToWorld, matO2W);
                mpb.SetMatrix(GaussianSplatRenderer.Props.MatrixWorldToObject, matW2O);
                mpb.SetVector(GaussianSplatRenderer.Props.VecScreenParams, screenPar);
                mpb.SetVector(GaussianSplatRenderer.Props.VecWorldSpaceCameraPos, camPos);

                mpb.SetFloat(GaussianSplatRenderer.Props.SplatScale, gs.m_SplatScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatOpacityScale, gs.m_OpacityScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatSize, settings.m_PointDisplaySize);
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOrder, gs.m_SHOrder);
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOnly, settings.m_SHOnly ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayIndex, settings.m_RenderMode == DebugRenderMode.DebugPointIndices ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayChunks, settings.m_RenderMode == DebugRenderMode.DebugChunkBounds ? 1 : 0);
                mpb.SetConstantBuffer(GaussianSplatRenderer.Props.SplatGlobalUniforms, m_GlobalUniforms, 0, m_GlobalUniforms.stride);

                int indexCount = 6;
                int instanceCount = gs.splatCount;

                // Perform octree culling if enabled
                if (settings.m_EnableOctreeCulling)
                {
                    int visibleCount = gs.PerformOctreeCulling(cam);
                    if (visibleCount > 0)
                    {
                        instanceCount = visibleCount;
                        // Update material property block with culling results
                        gs.SetAssetDataOnMaterial(mpb);
                    }
                    else
                    {
                        // Skip rendering if no splats are visible
                        continue;
                    }
                }

                MeshTopology topology = MeshTopology.Triangles;
                if (settings.m_RenderMode is DebugRenderMode.DebugBoxes or DebugRenderMode.DebugChunkBounds)
                    indexCount = 36;
                if (settings.m_RenderMode == DebugRenderMode.DebugChunkBounds)
                    instanceCount = gs.m_GpuChunksValid ? gs.m_GpuChunks.count : 0;

                cmb.BeginSample(s_ProfDraw);
                cmb.DrawProcedural(m_CubeIndexBuffer, matrix, displayMat, 0, topology, indexCount, instanceCount, mpb);
                cmb.EndSample(s_ProfDraw);
                
                // Store current matrix as previous for next frame
                gs.m_PrevMatrixMV = currentMatMV;
            }
        }

        // cube indices, most often we use only the first quad
        void EnsureCubeIndexBuffer()
        {
            if (m_CubeIndexBuffer != null)
                return;
            m_CubeIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
            m_CubeIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                4, 6, 5, 5, 6, 7,
                0, 2, 4, 4, 2, 6,
                1, 5, 3, 5, 7, 3,
                0, 4, 1, 4, 5, 1,
                2, 3, 6, 3, 7, 6
            });
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal void EnsureMaterials()
        {
            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            if (m_MatSplats == null && settings.resourcesFound)
            {
                m_MatSplats = new Material(settings.shaderSplats) {name = "GaussianSplats"};
                m_MatComposite = new Material(settings.shaderComposite) {name = "GaussianClearDstAlpha"};
                m_MatDebugPoints = new Material(settings.shaderDebugPoints) {name = "GaussianDebugPoints"};
                m_MatDebugBoxes = new Material(settings.shaderDebugBoxes) {name = "GaussianDebugBoxes"};
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        // ReSharper disable once UnusedMethodReturnValue.Global - used by HDRP/URP features that are not always compiled
        public CommandBuffer InitialClearCmdBuffer(Camera cam)
        {
            m_CommandBuffer ??= new CommandBuffer {name = "RenderGaussianSplats"};
            if (GraphicsSettings.currentRenderPipeline == null && cam != null && !m_CameraCommandBuffersDone.Contains(cam))
            {
                cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                m_CameraCommandBuffersDone.Add(cam);
            }

            // get render target for all splats
            m_CommandBuffer.Clear();
            return m_CommandBuffer;
        }

        void OnPreCullCamera(Camera cam)
        {
            if (!GatherSplatsForCamera(cam))
                return;

            EnsureMaterials();
            var matComposite = m_MatComposite;
            if (!matComposite)
                return;

            InitialClearCmdBuffer(cam);

            // We only need this to determine whether we're rendering into backbuffer or not. However, detection this
            // way only works in BiRP so only do it here.
            m_CommandBuffer.SetGlobalTexture(GaussianSplatRenderer.Props.CameraTargetTexture,
                BuiltinRenderTextureType.CameraTarget);

            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            if (!settings.isDebugRender)
            {
                // Set up render targets for temporal filtering - ensure WebGPU compatibility
                // Use ARGB32 for color target (matches FragOut.col : SV_Target0 - 4 components)
                // Use RG32 for motion target (matches FragOut.motion : SV_Target1 - 2 components) 
                // Create explicit descriptors so we can control sRGB/linear formats and avoid implicit gamma conversion
                int rtW = cam.pixelWidth;
                int rtH = cam.pixelHeight;
                // Choose a color format that matches the active color space: use sRGB format when in Gamma, linear format when in Linear.
                GraphicsFormat colorGfxFormat = QualitySettings.activeColorSpace == ColorSpace.Linear
                    ? GraphicsFormat.R8G8B8A8_UNorm
                    : GraphicsFormat.R8G8B8A8_SRGB;
                // Motion needs a linear floating format (no sRGB) - use RGBA format for WebGPU compatibility
                GraphicsFormat motionGfxFormat = GraphicsFormat.R16G16B16A16_SFloat;

                var descColor = new RenderTextureDescriptor(rtW, rtH, colorGfxFormat, 0) { msaaSamples = 1, useMipMap = false, autoGenerateMips = false };
                var descMotion = new RenderTextureDescriptor(rtW, rtH, motionGfxFormat, 0) { msaaSamples = 1, useMipMap = false, autoGenerateMips = false };

                m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, descColor, FilterMode.Point);
                m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatMotionRT, descMotion, FilterMode.Point);
                m_CommandBuffer.SetRenderTarget(new RenderTargetIdentifier[] { GaussianSplatRenderer.Props.GaussianSplatRT, GaussianSplatRenderer.Props.GaussianSplatMotionRT }, BuiltinRenderTextureType.CurrentActive);
                m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);
            }

            // add sorting, view calc and drawing commands for all splat objects
            SortAllSplats(cam, m_CommandBuffer);
            // View data calculation is now done in vertex shader
            RenderAllSplats(cam, m_CommandBuffer);

            // compose - with temporal filtering if enabled
            if (!settings.isDebugRender)
            {
                m_CommandBuffer.BeginSample(s_ProfCompose);
                if (settings.m_TemporalFilter != TemporalFilter.None)
                {
                    m_TemporalFilter ??= new GaussianSplatTemporalFilter();
                    m_TemporalFilter.Render(m_CommandBuffer, cam, matComposite, 1,
                        GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CameraTarget,
                        cam.pixelWidth, cam.pixelHeight,
                        settings.m_FrameInfluence, settings.m_VarianceClampScale,
                        GaussianSplatRenderer.Props.GaussianSplatMotionRT);
                }
                else
                {
                    m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
                    m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
                }
                m_CommandBuffer.EndSample(s_ProfCompose);
                m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
                m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatMotionRT);
            }
        }
    }

    [ExecuteInEditMode]
    public class GaussianSplatRenderer : MonoBehaviour
    {
        public GaussianSplatAsset m_Asset;

        [Tooltip("Rendering order compared to other splats. Within same order splats are sorted by distance. Higher order splats render 'on top of' lower order splats.")]
        public int m_RenderOrder;
        [Range(0.1f, 2.0f)] [Tooltip("Additional scaling factor for the splats")]
        public float m_SplatScale = 1.0f;
        [Range(0.05f, 20.0f)]
        [Tooltip("Additional scaling factor for opacity")]
        public float m_OpacityScale = 1.0f;
        [Range(0, 3)] [Tooltip("Spherical Harmonics order to use")]
        public int m_SHOrder = 3;

        // Static dummy buffer for WebGPU compatibility - ensures _SplatIndexMap is always bound
        static GraphicsBuffer s_DummyIndexBuffer;

        int m_SplatCount; // initially same as asset splat count
        GraphicsBuffer m_GpuPosData;
        GraphicsBuffer m_GpuOtherData;
        GraphicsBuffer m_GpuSHData;
        Texture m_GpuColorData;
        internal GraphicsBuffer m_GpuChunks;
        internal bool m_GpuChunksValid;
        internal int m_FrameCounter;
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;
        internal Matrix4x4 m_PrevMatrixMV = Matrix4x4.identity;
        bool m_Registered;

        // Octree culling system
        internal GaussianSplatOctree m_Octree;
        int m_LastCullingFrame = -1;
        internal bool m_OctreeBuilt;

        internal static class Props
        {
            public static readonly int SrcBlend = Shader.PropertyToID("_SrcBlend");
            public static readonly int DstBlend = Shader.PropertyToID("_DstBlend");
            public static readonly int ZWrite = Shader.PropertyToID("_ZWrite");
            public static readonly int SplatGlobalUniforms = Shader.PropertyToID("SplatGlobalUniforms");
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatOther = Shader.PropertyToID("_SplatOther");
            public static readonly int SplatSH = Shader.PropertyToID("_SplatSH");
            public static readonly int SplatColor = Shader.PropertyToID("_SplatColor");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatViewData = Shader.PropertyToID("_SplatViewData");
            public static readonly int SplatScale = Shader.PropertyToID("_SplatScale");
            public static readonly int SplatOpacityScale = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int SplatSize = Shader.PropertyToID("_SplatSize");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int SHOrder = Shader.PropertyToID("_SHOrder");
            public static readonly int SHOnly = Shader.PropertyToID("_SHOnly");
            public static readonly int DisplayIndex = Shader.PropertyToID("_DisplayIndex");
            public static readonly int DisplayChunks = Shader.PropertyToID("_DisplayChunks");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int GaussianSplatMotionRT = Shader.PropertyToID("_GaussianSplatMotionRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int PrevMatrixMV = Shader.PropertyToID("_PrevMatrixMV");
            public static readonly int PrevMatrixV = Shader.PropertyToID("_PrevMatrixV");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int VecScreenParams = Shader.PropertyToID("_VecScreenParams");
            public static readonly int VecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
            public static readonly int CameraTargetTexture = Shader.PropertyToID("_CameraTargetTexture");
            public static readonly int SplatIndexMap = Shader.PropertyToID("_SplatIndexMap");
            public static readonly int UseIndexMapping = Shader.PropertyToID("_UseIndexMapping");
        }



        public GaussianSplatAsset asset => m_Asset;
        public int splatCount => m_SplatCount;
        
        // Octree culling properties for editor access
        public bool octreeBuilt => m_OctreeBuilt;
        public GaussianSplatOctree octree => m_Octree;

        enum KernelIndices
        {
            // Keep only essential kernels - editing support removed
        }

        public bool HasValidAsset =>
            m_Asset != null &&
            m_Asset.splatCount > 0 &&
            m_Asset.formatVersion == GaussianSplatAsset.kCurrentVersion &&
            m_Asset.posData != null &&
            m_Asset.otherData != null &&
            m_Asset.shData != null &&
            m_Asset.colorData != null;
        public bool HasValidRenderSetup => m_GpuPosData != null && m_GpuOtherData != null && m_GpuChunks != null;

        void CreateResourcesForAsset()
        {
            if (!HasValidAsset)
                return;

            m_SplatCount = asset.splatCount;
            // For WebGL compatibility, use Vertex target instead of Raw
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, (int) (asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, (int) (asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, (int) (asset.shData.dataSize / 4), 4) { name = "GaussianSHData" };
            m_GpuSHData.SetData(asset.shData.GetData<uint>());
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            // For WebGL compatibility, use simpler texture creation flags
            var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.None) { name = "GaussianColorData" };
            tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
            tex.Apply(false, true);
            m_GpuColorData = tex;
            if (asset.chunkData != null && asset.chunkData.dataSize != 0)
            {
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Vertex,
                    (int) (asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
                m_GpuChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, 1,
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunksValid = false;
            }

            // Build octree for culling if enabled
            BuildOctreeForCulling();
        }

        bool resourcesAreSetUp => GaussianSplatSettings.instance.resourcesFound;

        public void EnsureSorterAndRegister()
        {
            if (!m_Registered && resourcesAreSetUp)
            {
                GaussianSplatRenderSystem.instance.RegisterSplat(this);
                m_Registered = true;
            }
        }

        public void OnEnable()
        {
            m_FrameCounter = 0;
            if (!resourcesAreSetUp)
                return;

            EnsureSorterAndRegister();
            CreateResourcesForAsset();
            // Update the MaterialPropertyBlock with asset data after creating resources
            UpdateAssetDataInRenderSystem();
        }



        internal void SetAssetDataOnMaterial(MaterialPropertyBlock mat)
        {
            if (!HasValidRenderSetup)
                return;
                
            mat.SetBuffer(Props.SplatPos, m_GpuPosData);
            mat.SetBuffer(Props.SplatOther, m_GpuOtherData);
            mat.SetBuffer(Props.SplatSH, m_GpuSHData);
            mat.SetTexture(Props.SplatColor, m_GpuColorData);
            mat.SetBuffer(Props.SplatChunks, m_GpuChunks);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            mat.SetInteger(Props.SplatFormat, (int)format);
            mat.SetInteger(Props.SplatCount, m_SplatCount);
            mat.SetInteger(Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);

            // Set octree culling properties
            var settings = GaussianSplatSettings.instance;
            bool useOctreeCulling = settings.m_EnableOctreeCulling && m_OctreeBuilt && m_Octree != null;
            mat.SetInteger(Props.UseIndexMapping, useOctreeCulling ? 1 : 0);
            
            // Always bind an index buffer for WebGPU compatibility
            if (useOctreeCulling && m_Octree.visibleIndicesBuffer != null)
            {
                mat.SetBuffer(Props.SplatIndexMap, m_Octree.visibleIndicesBuffer);
            }
            else
            {
                // Bind dummy buffer when not using octree culling to satisfy WebGPU requirements
                EnsureDummyIndexBuffer();
                mat.SetBuffer(Props.SplatIndexMap, s_DummyIndexBuffer);
            }
        }

        static void EnsureDummyIndexBuffer()
        {
            if (s_DummyIndexBuffer == null)
            {
                s_DummyIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, 4) { name = "GaussianDummyIndexMap" };
                // Fill with dummy data - just index 0
                s_DummyIndexBuffer.SetData(new uint[] { 0 });
            }
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        public static void DisposeDummyBuffers()
        {
            DisposeBuffer(ref s_DummyIndexBuffer);
        }

        void DisposeResourcesForAsset()
        {
            DestroyImmediate(m_GpuColorData);

            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuSHData);
            DisposeBuffer(ref m_GpuChunks);

            // Dispose octree
            m_Octree?.Dispose();
            m_Octree = null;
            m_OctreeBuilt = false;
            m_LastCullingFrame = -1;

            m_SplatCount = 0;
            m_GpuChunksValid = false;
        }

        public void OnDisable()
        {
            DisposeResourcesForAsset();
            GaussianSplatRenderSystem.instance.UnregisterSplat(this);
            m_Registered = false;
        }

        public void Update()
        {
            var curHash = m_Asset ? m_Asset.dataHash : new Hash128();
            if (m_PrevAsset != m_Asset || m_PrevHash != curHash)
            {
                m_PrevAsset = m_Asset;
                m_PrevHash = curHash;
                if (resourcesAreSetUp)
                {
                    DisposeResourcesForAsset();
                    CreateResourcesForAsset();
                    // Update the MaterialPropertyBlock with new asset data
                    UpdateAssetDataInRenderSystem();
                }
                else
                {
                    Debug.LogError($"{nameof(GaussianSplatRenderer)} component is not set up correctly (Resource references are missing), or platform does not support compute shaders");
                }
            }
        }

        void UpdateAssetDataInRenderSystem()
        {
            if (m_Registered)
            {
                GaussianSplatRenderSystem.instance.UpdateSplatAssetData(this);
            }
        }
// Draw octree leaf bounds in the Scene view for debugging. Call is automatic via OnDrawGizmos.
        public void OnDrawGizmos()
        {
            if (!m_OctreeBuilt || m_Octree == null)
                return;

            // Choose color based on debug mode if available
            var settings = GaussianSplatSettings.instance;
            Color col = (settings != null && settings.isDebugRender) ? Color.cyan : Color.yellow;

            m_Octree.DrawLeafBoundsGizmos(col);
        }
        public void ActivateCamera(int index)
        {
            Camera mainCam = Camera.main;
            if (!mainCam)
                return;
            if (!m_Asset || m_Asset.cameras == null)
                return;

            var selfTr = transform;
            var camTr = mainCam.transform;
            var prevParent = camTr.parent;
            var cam = m_Asset.cameras[index];
            camTr.parent = selfTr;
            camTr.localPosition = cam.pos;
            camTr.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
            camTr.parent = prevParent;
            camTr.localScale = Vector3.one;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(camTr);
#endif
        }

        public int PerformOctreeCulling(Camera camera)
        {
            if (!m_OctreeBuilt || m_Octree == null)
                return m_SplatCount; // No culling, render all splats
            
            var settings = GaussianSplatSettings.instance;
            if (!settings.m_EnableOctreeCulling)
                return m_SplatCount;
                
            // Check if we need to update culling (every N frames)
            int currentFrame = Time.frameCount;
            if (m_LastCullingFrame >= 0 && (currentFrame - m_LastCullingFrame) < settings.m_OctreeCullingUpdateInterval)
            {
                return m_Octree.visibleSplatCount; // Use cached result
            }
            
            m_LastCullingFrame = currentFrame;
            
            // For alpha blend mode, use hierarchical sorting which does culling + sorting in one pass
            if (settings.m_Transparency == TransparencyMode.AlphaBlend)
            {
                using (GaussianSplatRenderSystem.s_ProfSort.Auto())
                {
                    m_Octree.SortVisibleSplatsByDepth(camera);
                    return m_Octree.visibleSplatCount;
                }
            }
            else
            {
                // For other modes, use regular culling only
                return m_Octree.CullFrustum(camera);
            }
        }

        void BuildOctreeForCulling()
        {
            var settings = GaussianSplatSettings.instance;
            if (!settings.m_EnableOctreeCulling)
            {
                m_OctreeBuilt = false;
                return;
            }

            try
            {
                Debug.Log($"Building octree for {name}: SplatCount={m_SplatCount}, Format={asset.posFormat}");
                
                // Extract splat positions from asset data
                var splatPositions = ExtractSplatPositions();
                if (splatPositions.Length == 0)
                {
                    Debug.LogWarning($"No splat positions extracted for {name}");
                    m_OctreeBuilt = false;
                    return;
                }

                // Calculate scene bounds (apply object transform to account for scale/rotation/translation)
                NativeArray<float3> worldSplatPositions = new NativeArray<float3>(splatPositions.Length, Allocator.Temp);
                try
                {
                    var tr = transform;
                    for (int i = 0; i < splatPositions.Length; i++)
                    {
                        var p = splatPositions[i];
                        var wp = tr.TransformPoint(new Vector3(p.x, p.y, p.z));
                        worldSplatPositions[i] = new float3(wp.x, wp.y, wp.z);
                    }

                    var bounds = CalculateSplatBounds(worldSplatPositions);
                    Debug.Log($"Scene bounds for {name} (world-space): {bounds}");

                    // Initialize and build octree using world-space positions
                    m_Octree ??= new GaussianSplatOctree();
                    m_Octree.Initialize(settings.m_OctreeMaxDepth, settings.m_OctreeMaxSplatsPerLeaf);
                    m_Octree.Build(worldSplatPositions, bounds, settings.m_OctreeSplatRatio);
                }
                finally
                {
                    splatPositions.Dispose();
                    worldSplatPositions.Dispose();
                }

                m_OctreeBuilt = true;

                // Log debug info
                m_Octree.GetDebugInfo(out int leafNodes, out int maxDepth, out int maxSplatsInLeaf);
                Debug.Log($"Gaussian Splat Octree built for {name}: {leafNodes} leaf nodes, max depth {maxDepth}, max splats per leaf {maxSplatsInLeaf}");
                
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to build octree for {name}: {e.Message}\nStack trace: {e.StackTrace}");
                m_OctreeBuilt = false;
            }
        }

        NativeArray<float3> ExtractSplatPositions()
        {
            if (!HasValidAsset)
                return new NativeArray<float3>();

            // Add validation
            if (m_SplatCount <= 0)
            {
                Debug.LogError($"Invalid splat count: {m_SplatCount}");
                return new NativeArray<float3>();
            }

            var positions = new NativeArray<float3>(m_SplatCount, Allocator.Temp);
            var posData = asset.posData.GetData<uint>();
            var chunkData = asset.chunkData?.GetData<GaussianSplatAsset.ChunkInfo>();
            
            int vectorSize = GaussianSplatAsset.GetVectorSize(asset.posFormat);
            
            // Calculate expected data size and validate
            long expectedDataSize = (long)m_SplatCount * vectorSize;
            long actualDataSize = posData.Length * 4; // posData is uint[], so 4 bytes per element
            
            if (expectedDataSize > actualDataSize)
            {
                Debug.LogError($"Position data size mismatch: expected {expectedDataSize} bytes, got {actualDataSize} bytes. " +
                             $"SplatCount={m_SplatCount}, VectorSize={vectorSize}, Format={asset.posFormat}");
                positions.Dispose();
                return new NativeArray<float3>();
            }
            
            Debug.Log($"Extracting {m_SplatCount} splat positions. Format: {asset.posFormat}, VectorSize: {vectorSize}, " +
                     $"PosData length: {posData.Length} uints ({posData.Length * 4} bytes)");
            
            for (int i = 0; i < m_SplatCount; i++)
            {
                positions[i] = DecodeSplatPosition(posData, chunkData, i, asset.posFormat, vectorSize);
            }
            
            return positions;
        }

        float3 DecodeSplatPosition(NativeArray<uint> posData, NativeArray<GaussianSplatAsset.ChunkInfo>? chunkData, int splatIndex, GaussianSplatAsset.VectorFormat format, int vectorSize)
        {
            // Calculate byte address for this splat's position data
            int byteAddr = splatIndex * vectorSize;
            int uintAddr = byteAddr / 4;
            
            // Check bounds to prevent out-of-range errors
            if (uintAddr >= posData.Length)
            {
                Debug.LogError($"Position data out of bounds: uintAddr={uintAddr}, posData.Length={posData.Length}, splatIndex={splatIndex}, vectorSize={vectorSize}");
                return float3.zero;
            }
            
            float3 position = float3.zero;
            
            switch (format)
            {
                case GaussianSplatAsset.VectorFormat.Float32:
                    // 3 consecutive float32 values - need to check we have enough data
                    if (uintAddr + 2 >= posData.Length)
                    {
                        Debug.LogError($"Float32 position data out of bounds: need {uintAddr + 2}, have {posData.Length}");
                        return float3.zero;
                    }
                    position.x = math.asfloat(posData[uintAddr]);
                    position.y = math.asfloat(posData[uintAddr + 1]);
                    position.z = math.asfloat(posData[uintAddr + 2]);
                    break;
                    
                case GaussianSplatAsset.VectorFormat.Norm16:
                    // Packed 16.16.16 format (6 bytes total, needs special handling)
                    if (uintAddr + 1 >= posData.Length)
                    {
                        Debug.LogError($"Norm16 position data out of bounds: need {uintAddr + 1}, have {posData.Length}");
                        return float3.zero;
                    }
                    {
                        uint val0 = posData[uintAddr];
                        uint val1 = posData[uintAddr + 1];
                        // Handle unaligned access
                        if ((byteAddr & 3) != 0)
                        {
                            val0 = (val0 >> 16) | ((val1 & 0xFFFF) << 16);
                            val1 >>= 16;
                        }
                        position.x = (val0 & 0xFFFF) / 65535.0f;
                        position.y = ((val0 >> 16) & 0xFFFF) / 65535.0f;
                        position.z = (val1 & 0xFFFF) / 65535.0f;
                    }
                    break;
                    
                case GaussianSplatAsset.VectorFormat.Norm11:
                    // Packed 11.10.11 format (32 bits total)
                    {
                        uint val = posData[uintAddr];
                        if ((byteAddr & 3) != 0)
                        {
                            if (uintAddr + 1 >= posData.Length)
                            {
                                Debug.LogError($"Norm11 position data out of bounds for unaligned access: need {uintAddr + 1}, have {posData.Length}");
                                return float3.zero;
                            }
                            uint val1 = posData[uintAddr + 1];
                            val = (val >> 16) | ((val1 & 0xFFFF) << 16);
                        }
                        position.x = (val & 2047) / 2047.0f;
                        position.y = ((val >> 11) & 1023) / 1023.0f;
                        position.z = ((val >> 21) & 2047) / 2047.0f;
                    }
                    break;
                    
                case GaussianSplatAsset.VectorFormat.Norm6:
                    // Packed 6.5.5 format (16 bits total)
                    {

                        uint val = LoadUShortFromByteAddr(posData, byteAddr);
                        position.x = (val & 63) / 63.0f;
                        position.y = ((val >> 6) & 31) / 31.0f;
                        position.z = ((val >> 11) & 31) / 31.0f;
                    }
                    break;
            }
            
            // Apply chunk-relative positioning if chunk data exists
            if (chunkData.HasValue && chunkData.Value.IsCreated && chunkData.Value.Length > 0)
            {
                int chunkIndex = splatIndex / GaussianSplatAsset.kChunkSize;
                if (chunkIndex < chunkData.Value.Length)
                {
                    var chunk = chunkData.Value[chunkIndex];
                    // Convert chunk bounds to world space
                    position.x = math.lerp(chunk.posX.x, chunk.posX.y, position.x);
                    position.y = math.lerp(chunk.posY.x, chunk.posY.y, position.y);
                    position.z = math.lerp(chunk.posZ.x, chunk.posZ.y, position.z);
                }
            }
            else
            {
                // Use asset bounds
                var boundsMin = asset.boundsMin;
                var boundsMax = asset.boundsMax;
                position.x = math.lerp(boundsMin.x, boundsMax.x, position.x);
                position.y = math.lerp(boundsMin.y, boundsMax.y, position.y);
                position.z = math.lerp(boundsMin.z, boundsMax.z, position.z);
            }
            
            return position;
        }

        uint LoadUShortFromByteAddr(NativeArray<uint> data, int byteAddr)
        {
            int alignedAddr = byteAddr & ~0x3;
            int uintIndex = alignedAddr / 4;
            
            if (uintIndex >= data.Length)
            {
                Debug.LogError($"LoadUShortFromByteAddr out of bounds: uintIndex={uintIndex}, data.Length={data.Length}, byteAddr={byteAddr}");
                return 0;
            }
            
            uint val = data[uintIndex];
            if (byteAddr != alignedAddr)
                val >>= 16;
            return val & 0xFFFF;
        }

        Bounds CalculateSplatBounds(NativeArray<float3> positions)
        {
            if (positions.Length == 0)
                return new Bounds();
                
            float3 min = positions[0];
            float3 max = positions[0];
            
            for (int i = 1; i < positions.Length; i++)
            {
                min = math.min(min, positions[i]);
                max = math.max(max, positions[i]);
            }
            
            // Add some padding to avoid edge cases
            float3 padding = (max - min) * 0.01f;
            min -= padding;
            max += padding;
            
            return new Bounds((max + min) * 0.5f, max - min);
        }
    }
}