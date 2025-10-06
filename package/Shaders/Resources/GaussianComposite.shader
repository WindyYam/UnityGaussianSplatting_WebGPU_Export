// SPDX-License-Identifier: MIT
Shader "Hidden/Gaussian Splatting/Composite"
{
CGINCLUDE
#pragma vertex vert
#pragma fragment frag

#include "UnityCG.cginc"

struct v2f
{
    float4 vertex : SV_POSITION;
    float2 uv : TEXCOORD0;
};

v2f vert (uint vtxID : SV_VertexID)
{
    v2f o;
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
	o.vertex = float4(quadPos, 1, 1);
    o.uv = quadPos * 0.5 + 0.5;
    o.uv.y = 1 - o.uv.y;
    return o;
}
ENDCG
    SubShader
    {
        ZWrite Off
        ZTest Always
        Cull Off

        // Composite rendered gaussian splats onto regularly rendered scene.
        // Splats are rendered in sRGB, effectively, so need to decode that.
        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha

CGPROGRAM
Texture2D _GaussianSplatRT;
SamplerState sampler_GaussianSplatRT;
half4 frag (v2f i) : SV_Target
{
    // Use UV sampling (correct for fullscreen quad). If the RT is sRGB, Unity will decode automatically.
    half4 col = _GaussianSplatRT.Sample(sampler_GaussianSplatRT, i.uv);
    // If your RT is NOT marked sRGB, then decode manually:
    col.rgb = GammaToLinearSpace(col.rgb);
    //col.a = saturate(col.a * 1.5);
    return col;
}
ENDCG
        }

        // Do TAA-like temporal filter for gaussian splats
        Pass
        {
            Blend Off

CGPROGRAM
// Very Low: #define TAA_YCOCG 0, params 0,0,0
// Low: #define TAA_YCOCG 0, params 0,1,1
// Medium: params 2,2,1
// High: params 2,2,2
#include "TemporalFilter.hlsl"

half4 frag (v2f i) : SV_Target
{
    return DoTemporalAA(i.uv, 2, 2, 2);
}
ENDCG
        }
    }
}
