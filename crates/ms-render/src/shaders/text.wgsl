// On-screen text (text.rs): textured glyph quads drawn onto the display
// target after the composite.
//
// Positions arrive in clip space (converted from target pixels on the CPU).
// The atlas is R8Unorm, 1 where a glyph pixel is set, sampled with nearest
// filtering; quads are whole multiples of 8 px, so each glyph pixel covers an
// exact `scale` x `scale` block. Colour is straight alpha; the pipeline blends
// SrcAlpha / OneMinusSrcAlpha (alpha: One / OneMinusSrcAlpha).

@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let coverage = textureSample(atlas, atlas_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
