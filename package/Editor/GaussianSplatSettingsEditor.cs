// SPDX-License-Identifier: MIT

using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    [CustomEditor(typeof(GaussianSplatSettings))]
    [CanEditMultipleObjects]
    public class GaussianSplatSettingsEditor : UnityEditor.Editor
    {
        SerializedProperty m_Transparency;
        SerializedProperty m_TemporalFilter;
        SerializedProperty m_FrameInfluence;
        SerializedProperty m_VarianceClampScale;
        SerializedProperty m_RenderMode;
        SerializedProperty m_PointDisplaySize;
        SerializedProperty m_SHOnly;
        
        // Octree culling properties
        SerializedProperty m_EnableOctreeCulling;
        SerializedProperty m_OctreeMaxDepth;
        SerializedProperty m_OctreeMaxSplatsPerLeaf;
        SerializedProperty m_OctreeCullingUpdateInterval;
        SerializedProperty m_OctreeSplatRatio;

        public void OnEnable()
        {
            m_Transparency = serializedObject.FindProperty("m_Transparency");
            m_TemporalFilter = serializedObject.FindProperty("m_TemporalFilter");
            m_FrameInfluence = serializedObject.FindProperty("m_FrameInfluence");
            m_VarianceClampScale = serializedObject.FindProperty("m_VarianceClampScale");
            m_RenderMode = serializedObject.FindProperty("m_RenderMode");
            m_PointDisplaySize = serializedObject.FindProperty("m_PointDisplaySize");
            m_SHOnly = serializedObject.FindProperty("m_SHOnly");
            
            // Octree culling
            m_EnableOctreeCulling = serializedObject.FindProperty("m_EnableOctreeCulling");
            m_OctreeMaxDepth = serializedObject.FindProperty("m_OctreeMaxDepth");
            m_OctreeMaxSplatsPerLeaf = serializedObject.FindProperty("m_OctreeMaxSplatsPerLeaf");
            m_OctreeCullingUpdateInterval = serializedObject.FindProperty("m_OctreeCullingUpdateInterval");
            m_OctreeSplatRatio = serializedObject.FindProperty("m_OctreeSplatRatio");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.Space();
            EditorGUILayout.PropertyField(m_Transparency);
            
            EditorGUILayout.PropertyField(m_TemporalFilter);
            if (m_TemporalFilter.intValue != (int)TemporalFilter.None)
            {
                EditorGUILayout.PropertyField(m_FrameInfluence);
                EditorGUILayout.PropertyField(m_VarianceClampScale);
            }

            EditorGUILayout.Space();
            GUILayout.Label("Debugging Tweaks", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(m_RenderMode);
            if (m_RenderMode.intValue is (int)DebugRenderMode.DebugPoints or (int)DebugRenderMode.DebugPointIndices)
                EditorGUILayout.PropertyField(m_PointDisplaySize);
            EditorGUILayout.PropertyField(m_SHOnly);

            EditorGUILayout.Space();
            GUILayout.Label("Octree Culling", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(m_EnableOctreeCulling);
            if (m_EnableOctreeCulling.boolValue)
            {
                EditorGUILayout.PropertyField(m_OctreeMaxDepth);
                EditorGUILayout.PropertyField(m_OctreeMaxSplatsPerLeaf);
                EditorGUILayout.PropertyField(m_OctreeCullingUpdateInterval);
                EditorGUILayout.PropertyField(m_OctreeSplatRatio);
                
                // Display octree statistics
                if (Application.isPlaying)
                {
                    ShowOctreeStatistics();
                }
            }

            serializedObject.ApplyModifiedProperties();
        }
        
        void ShowOctreeStatistics()
        {
            EditorGUILayout.Space();
            GUILayout.Label("Octree Statistics (Runtime)", EditorStyles.miniBoldLabel);
            
            var renderers = FindObjectsByType<GaussianSplatRenderer>(FindObjectsSortMode.None);
            if (renderers.Length == 0)
            {
                EditorGUILayout.LabelField("No Gaussian Splat Renderers found");
                return;
            }
            
            foreach (var renderer in renderers)
            {
                if (renderer.isActiveAndEnabled && renderer.HasValidAsset)
                {
                    EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                    EditorGUILayout.LabelField($"Renderer: {renderer.name}", EditorStyles.boldLabel);
                    
                    if (renderer.octreeBuilt && renderer.octree != null)
                    {
                        renderer.octree.GetDebugInfo(out int leafNodes, out int maxDepth, out int maxSplatsInLeaf);
                        EditorGUILayout.LabelField($"Total Splats: {renderer.splatCount:N0}");
                        EditorGUILayout.LabelField($"Visible Splats: {renderer.octree.visibleSplatCount:N0}");
                        EditorGUILayout.LabelField($"Culling Ratio: {(1.0f - (float)renderer.octree.visibleSplatCount / renderer.splatCount) * 100:F1}%");
                        EditorGUILayout.LabelField($"Octree Nodes: {renderer.octree.nodeCount:N0}");
                        EditorGUILayout.LabelField($"Leaf Nodes: {leafNodes:N0}");
                        EditorGUILayout.LabelField($"Max Depth: {maxDepth}");
                        EditorGUILayout.LabelField($"Max Splats/Leaf: {maxSplatsInLeaf:N0}");
                    }
                    else
                    {
                        EditorGUILayout.LabelField("Octree not built");
                    }
                    
                    EditorGUILayout.EndVertical();
                }
            }
        }
    }
}