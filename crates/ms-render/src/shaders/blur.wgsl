// One pass of the separable 9-tap Gaussian blur (spec §6.1). The weights
// `W` (W[0] centre, W[i] at offsets +-i) are prepended by blur.rs, which
// documents their values.
//
// The target is the blur level (or its scratch texture); `tap_step.xy` is one
// target texel along the blur axis, in uv. The horizontal pass reads the
// previous level (twice the size), so its bilinear taps at the target's
// texel centres also average 2x2 source texels: the downsample. Values are
// stored raw; `GetBlurN` applies blur_min/blur_max.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> tap_step: vec4<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle covering the target: (-1,-1), (3,-1), (-1,3).
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var out: VsOut;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    // v = 0 at the top (texture space).
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

// The two taps at +-i target texels.
fn pair(uv: vec2<f32>, i: f32) -> vec4<f32> {
    let offset = tap_step.xy * i;
    return textureSampleLevel(src, src_sampler, uv + offset, 0.0)
        + textureSampleLevel(src, src_sampler, uv - offset, 0.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, src_sampler, in.uv, 0.0) * W[0]
        + pair(in.uv, 1.0) * W[1]
        + pair(in.uv, 2.0) * W[2]
        + pair(in.uv, 3.0) * W[3]
        + pair(in.uv, 4.0) * W[4];
}
