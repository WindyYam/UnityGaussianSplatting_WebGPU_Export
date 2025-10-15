// SPDX-License-Identifier: MIT

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Native interface for WASM threading support for splat sorting.
    /// Provides background threading for WebGPU platform where Unity's Task system is not supported.
    /// </summary>
    public static class NativeSorting
    {
        // Job handle for tracking native sort operations
        public struct SortJobHandle
        {
            internal int jobId;
            internal bool isValid;

            public bool IsCompleted => isValid && NativeSorting_IsJobCompleted(jobId) != 0;
            public bool IsValid => isValid;
        }

        // Platform-specific native function declarations
#if UNITY_WEBGL && !UNITY_EDITOR
        // WebGL/WASM platform - uses Emscripten __Internal
        [DllImport("__Internal")]
        private static extern unsafe int NativeSorting_StartSortJob(
            int* splatIndices, int splatCount,
            float* positions, int positionCount,
            int* sortedIndices,
            float camX, float camY, float camZ);

        [DllImport("__Internal")]
        private static extern int NativeSorting_IsJobCompleted(int jobId);

        // Note: GetSortedIndices no longer needed - results written directly to output buffer

        [DllImport("__Internal")]
        private static extern void NativeSorting_CleanupJob(int jobId);

        [DllImport("__Internal")]
        private static extern int NativeSorting_GetWorkerCount();

        [DllImport("__Internal")]
        private static extern void NativeSorting_Initialize(int workerThreads);

        [DllImport("__Internal")]
        private static extern void NativeSorting_Shutdown();

#elif UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        // Windows platform - uses native DLL
        [DllImport("NativeSorting")]
        private static extern unsafe int NativeSorting_StartSortJob(
            int* splatIndices, int splatCount,
            float* positions, int positionCount,
            int* sortedIndices,
            float camX, float camY, float camZ);

        [DllImport("NativeSorting")]
        private static extern int NativeSorting_IsJobCompleted(int jobId);

        // Note: GetSortedIndices no longer needed - results written directly to output buffer

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_CleanupJob(int jobId);

        [DllImport("NativeSorting")]
        private static extern int NativeSorting_GetWorkerCount();

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_Initialize(int workerThreads);

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_Shutdown();

#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        // macOS platform - uses dylib
        [DllImport("NativeSorting")]
        private static extern unsafe int NativeSorting_StartSortJob(
            int* splatIndices, int splatCount,
            float* positions, int positionCount,
            int* sortedIndices,
            float camX, float camY, float camZ);

        [DllImport("NativeSorting")]
        private static extern int NativeSorting_IsJobCompleted(int jobId);

        // Note: GetSortedIndices no longer needed - results written directly to output buffer

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_CleanupJob(int jobId);

        [DllImport("NativeSorting")]
        private static extern int NativeSorting_GetWorkerCount();

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_Initialize(int workerThreads);

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_Shutdown();

#elif UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
        // Linux platform - uses shared library
        [DllImport("NativeSorting")]
        private static extern unsafe int NativeSorting_StartSortJob(
            int* splatIndices, int splatCount,
            float* positions, int positionCount,
            int* sortedIndices,
            float camX, float camY, float camZ);

        [DllImport("NativeSorting")]
        private static extern int NativeSorting_IsJobCompleted(int jobId);

        // Note: GetSortedIndices no longer needed - results written directly to output buffer

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_CleanupJob(int jobId);

        [DllImport("NativeSorting")]
        private static extern int NativeSorting_GetWorkerCount();

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_Initialize(int workerThreads);

        [DllImport("NativeSorting")]
        private static extern void NativeSorting_Shutdown();

#else
        // Fallback for unsupported platforms - stub implementations
        private static unsafe int NativeSorting_StartSortJob(
            int* splatIndices, int splatCount,
            float* positions, int positionCount,
            int* sortedIndices,
            float camX, float camY, float camZ) => -1;

        private static int NativeSorting_IsJobCompleted(int jobId) => 0;
        // Note: GetSortedIndices no longer needed - results written directly to output buffer
        private static void NativeSorting_CleanupJob(int jobId) { }
        private static int NativeSorting_GetWorkerCount() => 1;
        private static void NativeSorting_Initialize(int workerThreads) { }
        private static void NativeSorting_Shutdown() { }
#endif

        // Platform detection and state
        private static bool s_IsSupportedPlatform = false;
        private static bool s_IsInitialized = false;

        static NativeSorting()
        {
            // Determine if current platform supports native threading
            s_IsSupportedPlatform = Application.platform == RuntimePlatform.WebGLPlayer ||
                                   Application.platform == RuntimePlatform.WindowsPlayer ||
                                   Application.platform == RuntimePlatform.OSXPlayer ||
                                   Application.platform == RuntimePlatform.LinuxPlayer ||
                                   Application.isEditor; // Include editor for testing
        }

        /// <summary>
        /// Initialize native sorting system with specified worker thread count.
        /// Supported on WebGL, Windows, macOS, and Linux platforms.
        /// </summary>
        public static void Initialize(int workerThreads = 4)
        {
            if (!s_IsSupportedPlatform || s_IsInitialized)
                return;

            try
            {
                NativeSorting_Initialize(workerThreads);
                s_IsInitialized = true;
                string platformName = GetPlatformName();
                Debug.Log($"NativeSorting: Initialized with {workerThreads} worker threads on {platformName} platform");
            }
            catch (DllNotFoundException dllEx)
            {
                string platformName = GetPlatformName();
                Debug.LogWarning($"NativeSorting: Native library not found on {platformName} platform. " +
                               $"Expected library: {GetExpectedLibraryName()}. " +
                               $"Falling back to Unity Task system. " +
                               $"To enable native threading, build and place the native library in Assets/Plugins/");
            }
            catch (EntryPointNotFoundException entryEx)
            {
                string platformName = GetPlatformName();
                Debug.LogError($"NativeSorting: Native library found but missing required functions on {platformName} platform. " +
                              $"Library may be outdated or incompatible: {entryEx.Message}");
            }
            catch (Exception e)
            {
                string platformName = GetPlatformName();
                Debug.LogError($"NativeSorting: Failed to initialize on {platformName} platform: {e.Message}");
            }
        }

        /// <summary>
        /// Shutdown native sorting system and cleanup resources.
        /// </summary>
        public static void Shutdown()
        {
            if (!s_IsSupportedPlatform || !s_IsInitialized)
                return;

            try
            {
                NativeSorting_Shutdown();
                s_IsInitialized = false;
                Debug.Log("NativeSorting: Shutdown completed");
            }
            catch (Exception e)
            {
                Debug.LogError($"NativeSorting: Error during shutdown: {e.Message}");
            }
        }

        /// <summary>
        /// Get the number of available worker threads from native side.
        /// </summary>
        public static int GetWorkerCount()
        {
            if (!s_IsSupportedPlatform || !s_IsInitialized)
                return 1;

            try
            {
                return NativeSorting_GetWorkerCount();
            }
            catch (Exception e)
            {
                Debug.LogError($"NativeSorting: Error getting worker count: {e.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Start a background sort job for the given splat indices.
        /// Results will be written directly to the sortedIndices array.
        /// </summary>
        public static SortJobHandle StartSortJob(
            NativeArray<int> splatIndices, 
            NativeArray<float3> positions, 
            NativeArray<int> sortedIndices,
            Vector3 cameraPosition)
        {
            if (!s_IsSupportedPlatform || !s_IsInitialized)
            {
                return new SortJobHandle { jobId = -1, isValid = false };
            }

            try
            {
                unsafe
                {
                    var splatPtr = (int*)splatIndices.GetUnsafeReadOnlyPtr();
                    var posPtr = (float*)positions.GetUnsafeReadOnlyPtr();
                    var sortedPtr = (int*)sortedIndices.GetUnsafePtr();
                    
                    int jobId = NativeSorting_StartSortJob(
                        splatPtr, splatIndices.Length,
                        posPtr, positions.Length,
                        sortedPtr,
                        cameraPosition.x, cameraPosition.y, cameraPosition.z);

                    if (jobId >= 0)
                    {
                        return new SortJobHandle { jobId = jobId, isValid = true };
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"NativeSorting: Error starting sort job: {e.Message}");
            }

            return new SortJobHandle { jobId = -1, isValid = false };
        }

        // Note: GetSortedIndices method removed - results are written directly to the output buffer passed to StartSortJob

        /// <summary>
        /// Cleanup a completed or cancelled job.
        /// </summary>
        public static void CleanupJob(SortJobHandle handle)
        {
            if (!s_IsSupportedPlatform || !s_IsInitialized || !handle.IsValid)
                return;

            try
            {
                NativeSorting_CleanupJob(handle.jobId);
            }
            catch (Exception e)
            {
                Debug.LogError($"NativeSorting: Error cleaning up job: {e.Message}");
            }
        }

        /// <summary>
        /// Check if native sorting is available on current platform.
        /// </summary>
        public static bool IsAvailable => s_IsSupportedPlatform && s_IsInitialized;

        /// <summary>
        /// Get a friendly name for the current platform.
        /// </summary>
        private static string GetPlatformName()
        {
            switch (Application.platform)
            {
                case RuntimePlatform.WebGLPlayer:
                    return "WebGL";
                case RuntimePlatform.WindowsPlayer:
                case RuntimePlatform.WindowsEditor:
                    return "Windows";
                case RuntimePlatform.OSXPlayer:
                case RuntimePlatform.OSXEditor:
                    return "macOS";
                case RuntimePlatform.LinuxPlayer:
                case RuntimePlatform.LinuxEditor:
                    return "Linux";
                default:
                    return Application.platform.ToString();
            }
        }

        /// <summary>
        /// Get the expected native library filename for the current platform.
        /// </summary>
        private static string GetExpectedLibraryName()
        {
            switch (Application.platform)
            {
                case RuntimePlatform.WebGLPlayer:
                    return "NativeSorting.jslib (compiled to WASM)";
                case RuntimePlatform.WindowsPlayer:
                case RuntimePlatform.WindowsEditor:
                    return "NativeSorting.dll";
                case RuntimePlatform.OSXPlayer:
                case RuntimePlatform.OSXEditor:
                    return "NativeSorting.dylib";
                case RuntimePlatform.LinuxPlayer:
                case RuntimePlatform.LinuxEditor:
                    return "libNativeSorting.so";
                default:
                    return "NativeSorting library";
            }
        }
    }
}
