// Built-in warp: draws the warp mesh, sampling the previous feedback buffer
// at each vertex's UV, and writes sample * decay * vertex colour.
//
// Group 0 matches the layout every warp shader uses (see warp.rs):
//   0 u: Uniforms, 1 sampler_main, 11 s_fw (linear wrap), 12 s_fc (linear clamp).
// Fragment alpha is the layer weight; the pipeline blends colour as
// src * srcAlpha + dst and leaves the target's alpha (cleared to 1) alone.

// Must match `Uniforms` in warp.rs field for field.
struct Uniforms {
    time: f32, fps: f32, frame: f32, progress: f32,
    bass: f32, mid: f32, treb: f32, beat: f32,
    bass_att: f32, mid_att: f32, treb_att: f32, beat_strength: f32,
    decay: f32, gamma: f32, echo_zoom: f32, echo_alpha: f32,
    echo_orient: f32, invert: f32, solarize: f32, brighten: f32,
    darken: f32, darken_center: f32, aspect_x: f32, aspect_y: f32,
    texsize: vec4<f32>,
    rand_frame: vec4<f32>,
    rand_preset: vec4<f32>,
    roam_cos: vec4<f32>,
    roam_sin: vec4<f32>,
    blur_min: vec4<f32>,
    blur_max: vec4<f32>,
    q: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sampler_main: texture_2d<f32>;
@group(0) @binding(11) var s_fw: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    // rgb = per-vertex colour, a = layer weight.
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

// NaN/Inf (exponent bits all ones) become 0, then clamp to [0, 64].
// WGSL has no isnan/isinf, so test the bits.
fn sanitise(v: vec3<f32>) -> vec3<f32> {
    let exponent = bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u);
    let bad = exponent == vec3<u32>(0x7f800000u);
    return clamp(select(v, vec3<f32>(0.0), bad), vec3<f32>(0.0), vec3<f32>(64.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let prev = textureSample(sampler_main, s_fw, in.uv).rgb;
    return vec4<f32>(sanitise(prev * u.decay * in.color.rgb), in.color.a);
}
