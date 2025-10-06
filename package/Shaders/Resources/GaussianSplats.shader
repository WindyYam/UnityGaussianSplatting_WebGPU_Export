// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats"
{
	Properties
	{
		_SrcBlend("Src Blend", Float) = 8 // OneMinusDstAlpha
		_DstBlend("Dst Blend", Float) = 1 // One
		_ZWrite("ZWrite", Float) = 0  // Off
	}
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite [_ZWrite]
            Blend [_SrcBlend] [_DstBlend]
            Cull Off
            
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
// Remove compute shader requirement for WebGL compatibility
// #pragma require compute

#include "GaussianSplatting.hlsl"

float _SplatScale;
float _SplatOpacityScale;
uint _SHOrder;
uint _SHOnly;
float4x4 _MatrixMV;
float4x4 _PrevMatrixMV;
float4x4 _PrevMatrixV;
float4x4 _MatrixObjectToWorld;
float4x4 _MatrixWorldToObject;
float4 _VecScreenParams;
float4 _VecWorldSpaceCameraPos;
uint _UseIndexMapping; // 1 when using octree culling, 0 otherwise

struct v2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    uint idx : TEXCOORD1;
    float2 vel : TEXCOORD2; // NDC motion delta (current - previous)
    float4 vertex : SV_POSITION;
};

// reason for using a separate uniform buffer is DX12
// stale uniform variable bug in Unity 2022.3/6000.0 at least,
// "IN-99220 - DX12 stale render state issue with a sequence of compute shader & DrawProcedural calls"
cbuffer SplatGlobalUniforms // match struct SplatGlobalUniforms in C#
{
	uint sgu_transparencyMode;
	uint sgu_frameOffset;
	uint sgu_needMotionVectors; // Motion vector type, 0: None, 1: Simple(center), 2: Per vertex
	uint sgu_padding0; // padding to align cbuffer to 16 bytes
}

StructuredBuffer<uint> _SplatIndexMap; // optional mapping

// Helper function to decompose 2D covariance into screen-space axes
void DecomposeCovariance(float3 cov2d, out float2 v1, out float2 v2)
{
    // same as in antimatter15/splat
    float diag1 = cov2d.x, diag2 = cov2d.z, offDiag = cov2d.y;
    float mid = 0.5f * (diag1 + diag2);
    float radius = length(float2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    float2 diagVec = normalize(float2(offDiag, lambda1 - diag1));
    diagVec.y = -diagVec.y;
    float maxSize = 4096.0;
    v1 = min(sqrt(2.0 * lambda1), maxSize) * diagVec;
    v2 = min(sqrt(2.0 * lambda2), maxSize) * float2(diagVec.y, -diagVec.x);
}

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o = (v2f)0;
    uint realIdx = instID;
    if (_UseIndexMapping)
        realIdx = _SplatIndexMap[instID];

    o.idx = realIdx + sgu_frameOffset;
    SplatData splat = LoadSplatData(realIdx);
    
    // Transform to world space
    float3 centerWorldPos = mul(_MatrixObjectToWorld, float4(splat.pos, 1)).xyz;
    float4 centerClipPos = mul(UNITY_MATRIX_VP, float4(centerWorldPos, 1));
    
    // Check if behind camera, outside frustum, or outside near/far clip planes
    bool behindCam = centerClipPos.w <= 0;
    float2 ndc = centerClipPos.xy / centerClipPos.w;
    float ndcZ = centerClipPos.z / centerClipPos.w;
    bool outsideFrustum = abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0;
    bool outsideClipPlanes = ndcZ < 0.0 || ndcZ > 1.0;
    if (behindCam || outsideFrustum || outsideClipPlanes)
    {
        o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
        return o;
    }
    
    // Calculate 3D covariance matrix
    float3x3 splatRotScaleMat = CalcMatrixFromRotationScale(splat.rot, splat.scale);
    float3 cov3d0, cov3d1;
    CalcCovariance3D(splatRotScaleMat, cov3d0, cov3d1);
    float splatScale2 = _SplatScale * _SplatScale;
    cov3d0 *= splatScale2;
    cov3d1 *= splatScale2;
    
    // Project 3D covariance to 2D screen space
    float2 screenCenter2D;
    float3 cov2d = CalcCovariance2D(splat.pos, cov3d0, cov3d1, _MatrixMV, UNITY_MATRIX_P, _VecScreenParams, screenCenter2D);
    
    // Update clip position with corrected screen center
    float4 centerClipPosCorrected = centerClipPos;
    centerClipPosCorrected.xy = (screenCenter2D * 2/_VecScreenParams.xy - 1) * centerClipPos.w;
    
    // Decompose 2D covariance into screen-space axes
    float2 axis1, axis2;
    DecomposeCovariance(cov2d, axis1, axis2);
    
    // Calculate color using spherical harmonics
    float3 worldViewDir = _VecWorldSpaceCameraPos.xyz - centerWorldPos;
    float3 objViewDir = mul((float3x3)_MatrixWorldToObject, worldViewDir);
    objViewDir = normalize(objViewDir);
    half3 col = ShadeSH(splat.sh, objViewDir, _SHOrder, _SHOnly != 0);
    half opacity = splat.opacity * _SplatOpacityScale;
    
    o.col = half4(col, opacity);
    
    // Generate quad vertex
    uint idx = vtxID;
    float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
    quadPos *= 2;
    o.pos = quadPos;
    
    // Apply screen-space offset
    float2 deltaScreenPos = (quadPos.x * axis1 + quadPos.y * axis2) * 2 / _ScreenParams.xy;
    o.vertex = centerClipPosCorrected;
    o.vertex.xy += deltaScreenPos * centerClipPosCorrected.w;

    // Motion vectors - calculate from center position NDC difference when requested
    if (sgu_needMotionVectors == 1)
    {
        // Approximate the motion vector, calc the center movement only
        // Current center NDC (use corrected center position only, not per-vertex offset)
        float2 currentCenterNDC = centerClipPos.xy / centerClipPos.w;

        // If current center is outside of frustum, skip motion vector (set to zero)
        if (abs(currentCenterNDC.x) > 1.0 || abs(currentCenterNDC.y) > 1.0)
        {
            o.vel = float2(0.0, 0.0);
        }
        else
        { 
            // Previous center clip position: project world center with previous view and current projection
            float4 prevCenterClipPos = mul(UNITY_MATRIX_P, mul(_PrevMatrixV, float4(centerWorldPos, 1)));

            // Previous center NDC
            float2 prevCenterNDC = prevCenterClipPos.xy / prevCenterClipPos.w;
            // Motion vector is center NDC difference (approximation, avoids per-vertex offsets)
            o.vel = currentCenterNDC - prevCenterNDC;
            FlipMotionIfBackbuffer(o.vel);
        }
    }
    else if (sgu_needMotionVectors == 2)
    {
        // More precise calc but intensive
        float2 currentNDC = (o.vertex.xy / o.vertex.w);

        if (abs(currentNDC.x) > 1.0 || abs(currentNDC.y) > 1.0)
        {
            o.vel = float2(0.0, 0.0);
        }
        else
        { 
            // Calculate previous position for motion vectors
            // Project 3D covariance to 2D screen space using previous matrix
            float2 prevScreenCenter2D;
            float3 prevCov2d = CalcCovariance2D(splat.pos, cov3d0, cov3d1, _PrevMatrixMV, UNITY_MATRIX_P, _VecScreenParams, prevScreenCenter2D);

            // Decompose previous 2D covariance into screen-space axes
            float2 prevAxis1, prevAxis2;
            DecomposeCovariance(prevCov2d, prevAxis1, prevAxis2);

            // Calculate previous clip position (assuming same w as current since splats are static)
            float4 prevCenterClipPos = centerClipPos;
            prevCenterClipPos.xy = (prevScreenCenter2D * 2/_VecScreenParams.xy - 1) * centerClipPos.w;

            // Apply same quad offset to get previous vertex position
            float2 prevDeltaScreenPos = (quadPos.x * prevAxis1 + quadPos.y * prevAxis2) * 2 / _ScreenParams.xy;
            float4 prevVertex = prevCenterClipPos;
            prevVertex.xy += prevDeltaScreenPos * centerClipPos.w;

            // Calculate previous NDC position
            float2 prevNDC = (prevVertex.xy / prevVertex.w);
            o.vel = currentNDC - prevNDC;
            FlipMotionIfBackbuffer(o.vel);
        }
    }
    else
    {
        o.vel = float2(0.0, 0.0);
    }

	FlipProjectionIfBackbuffer(o.vertex);
    return o;
}

// Hash Functions for GPU Rendering
// https://jcgt.org/published/0009/03/02/
uint3 pcg3d16(uint3 v)
{
    v = v * 12829u + 47989u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

	v >>= 16u;
    return v;
}

struct FragOut { half4 col : SV_Target0; half4 motion : SV_Target1; };

FragOut frag (v2f i)
{
    FragOut o; o.col = 0; o.motion = half4(0, 0, 0, 0);
    float power = -dot(i.pos, i.pos);
    half alpha = exp(power);
    if (i.col.a >= 0)
    {
        alpha = saturate(alpha * i.col.a);
    }
    else
    {
        half3 selectedColor = half3(1,0,1);
        if (alpha > 7.0/255.0)
        {
            if (alpha < 10.0/255.0)
            {
                alpha = 1;
                i.col.rgb = selectedColor;
            }
            alpha = saturate(alpha + 0.3);
        }
        i.col.rgb = lerp(i.col.rgb, selectedColor, 0.5);
    }
    if (alpha < 1.0/255.0)
        discard;

    if (sgu_transparencyMode == 0)
    {
        uint3 coord = uint3(i.vertex.x, i.vertex.y, i.idx);
        uint3 hash = pcg3d16(coord);
        half cutoff = (hash.x & 0xFFFF) / 65535.0;
        if (alpha <= cutoff)
            discard;
        alpha = 1;
    }
    else if (sgu_transparencyMode == 1)
    {
        // Halftone ordered dither (4x4 Bayer) for stable stochastic-like transparency.
        // Use screen-space integer pixel coordinates and a small per-instance offset
        // to decorrelate neighboring splats. This is deterministic across frames,
        // reducing temporal flicker compared to random sampling.
        uint2 p = uint2(i.vertex.xy); // truncate to integer pixel coords
        uint bx = p.x & 3u;
        uint by = p.y & 3u;

        // Apply a small per-splat offset to the dither tile to reduce repeating artifacts.
        uint inst = i.idx;
        bx = (bx + (inst & 3u)) & 3u;
        by = (by + ((inst >> 2) & 3u)) & 3u;

        // 4x4 Bayer matrix values 0..15
        const uint bayer4[16] = {
            0u, 8u, 2u, 10u,
            12u,4u,14u,6u,
            3u,11u,1u,9u,
            15u,7u,13u,5u
        };
        uint t = bayer4[bx + (by << 2)];
        half cutoff = (t + 0.5) / 16.0;

        // Binary decision: keep pixel if alpha exceeds threshold, otherwise discard.
        if (alpha <= cutoff)
            discard;
        alpha = 1;
    }
    else // sgu_transparencyMode == 2 (AlphaBlend)
    {
        i.col.rgb *= alpha; // premultiply
        // This mode will require proper depth sorting for correct results
    }
    o.motion = half4(i.vel, 0, 0); // Use xy for motion vectors, zw unused for WebGPU compatibility
    o.col = half4(i.col.rgb, alpha);
    return o;
}
ENDCG
        }
    }
}
