// Overlays drawn into the feedback buffer: motion vectors, shapes, custom
// waves, the main waveform and the borders (draw.rs). One shader serves
// them all; only textured shapes use the texture.
//
// Positions are in MilkDrop's space ([0, 1], origin at the bottom left) and
// become clip space here. Colours arrive with straight alpha. The "normal"
// pipelines (blend One, OneMinusSrcAlpha) premultiply here; the additive
// ones (SrcAlpha, One) do not. Both keep the target's alpha.

// The previous feedback buffer (never the one being drawn into).
@group(0) @binding(0) var previous: texture_2d<f32>;
@group(0) @binding(1) var previous_sampler: sampler;

override ms_premultiply: bool = true;

struct VsIn {
    @location(0) pos: vec2<f32>,
    // Texture space (v = 0 at the top); textured shapes only.
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    // 1 for textured shapes, else 0.
    @location(3) textured: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) textured: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(in.pos * 2.0 - 1.0, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.textured = in.textured;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Sampled unconditionally (uniform control flow); untextured geometry
    // ignores the result.
    let texel = textureSample(previous, previous_sampler, in.uv).rgb;
    let rgb = in.color.rgb * mix(vec3<f32>(1.0), texel, in.textured);
    let a = in.color.a;
    if ms_premultiply {
        return vec4<f32>(rgb * a, a);
    }
    return vec4<f32>(rgb, a);
}
