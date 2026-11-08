// milksplat shader prelude (spec §6.5). Every warp and composite shader, the
// built-in ones included, is compiled as this prelude followed by:
//   - group 1 declarations for the preset's user textures (generated):
//       @group(1) @binding(2i)     var sampler_X: texture_2d<f32>;
//       @group(1) @binding(2i + 1) var sampler_X_s: sampler;
//   - the stage's vertex shader and fragment entry point;
//   - `fn user_warp(uv: vec2f, uv_orig: vec2f, rad: f32, ang: f32,
//     hue_shader: vec3f) -> vec3f { <body> }` or
//     `fn user_comp(uv: vec2f, rad: f32, ang: f32, hue_shader: vec3f)
//     -> vec3f { <body> }`, the body being the preset's shader source.
//
// Texture space has v = 0 at the top.

// Must match `Uniforms` in shaders.rs field for field (std140: every row is
// 16 bytes; the `uniforms_layout` test checks the offsets with naga).
struct Uniforms {
    time: f32, fps: f32, frame: f32, progress: f32,
    bass: f32, mid: f32, treb: f32, beat: f32,
    bass_att: f32, mid_att: f32, treb_att: f32, beat_strength: f32,
    decay: f32, gamma: f32, echo_zoom: f32, echo_alpha: f32,
    echo_orient: f32, invert: f32, solarize: f32, brighten: f32,
    darken: f32, darken_center: f32, aspect_x: f32, aspect_y: f32,
    // xy = size in pixels, zw = 1 / size.
    texsize: vec4<f32>,
    // New every tick.
    rand_frame: vec4<f32>,
    // Fixed per preset.
    rand_preset: vec4<f32>,
    roam_cos: vec4<f32>,
    roam_sin: vec4<f32>,
    // x, y, z = blur levels 1, 2, 3.
    blur_min: vec4<f32>,
    blur_max: vec4<f32>,
    // q1..q32, four per vec4.
    q: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// Warp: the previous frame. Composite: the latest frame.
@group(0) @binding(1) var sampler_main: texture_2d<f32>;
@group(0) @binding(2) var sampler_blur1: texture_2d<f32>;
@group(0) @binding(3) var sampler_blur2: texture_2d<f32>;
@group(0) @binding(4) var sampler_blur3: texture_2d<f32>;
@group(0) @binding(5) var sampler_noise_lq: texture_2d<f32>;
@group(0) @binding(6) var sampler_noise_mq: texture_2d<f32>;
@group(0) @binding(7) var sampler_noise_hq: texture_2d<f32>;
@group(0) @binding(8) var sampler_noisevol_lq: texture_3d<f32>;
@group(0) @binding(9) var sampler_noisevol_hq: texture_3d<f32>;
@group(0) @binding(10) var sampler_spectrum: texture_2d<f32>;
// Linear + wrap, linear + clamp, point + wrap, point + clamp.
@group(0) @binding(11) var s_fw: sampler;
@group(0) @binding(12) var s_fc: sampler;
@group(0) @binding(13) var s_pw: sampler;
@group(0) @binding(14) var s_pc: sampler;

// The main image, linear filtering, wrap addressing.
fn GetMain(uv: vec2<f32>) -> vec3<f32> {
    return textureSample(sampler_main, s_fw, uv).rgb;
}

// The main image, point sampling, wrap addressing.
fn GetPixel(uv: vec2<f32>) -> vec3<f32> {
    return textureSample(sampler_main, s_pw, uv).rgb;
}

// Blur level N, mapped back from its stored [0, 1] range to
// [blur_min, blur_max] (linear filtering, clamp addressing).
fn GetBlur1(uv: vec2<f32>) -> vec3<f32> {
    return u.blur_min.x + textureSample(sampler_blur1, s_fc, uv).rgb * (u.blur_max.x - u.blur_min.x);
}

fn GetBlur2(uv: vec2<f32>) -> vec3<f32> {
    return u.blur_min.y + textureSample(sampler_blur2, s_fc, uv).rgb * (u.blur_max.y - u.blur_min.y);
}

fn GetBlur3(uv: vec2<f32>) -> vec3<f32> {
    return u.blur_min.z + textureSample(sampler_blur3, s_fc, uv).rgb * (u.blur_max.z - u.blur_min.z);
}

// NaN/Inf (exponent bits all ones) become 0, then clamp to [0, 64].
// WGSL has no isnan/isinf, so test the bits.
fn ms_sanitise(v: vec3<f32>) -> vec3<f32> {
    let exponent = bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u);
    let bad = exponent == vec3<u32>(0x7f800000u);
    return clamp(select(v, vec3<f32>(0.0), bad), vec3<f32>(0.0), vec3<f32>(64.0));
}

// Aspect-corrected offset from the centre for a texture-space position:
// x right, y up, about [-1, 1] on the short axis.
fn ms_centre_offset(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * vec2<f32>(u.aspect_x, u.aspect_y);
}

// `rad`: distance from the centre, 0 at the centre and 1 at the corners of a
// square image.
fn ms_rad(uv: vec2<f32>) -> f32 {
    return length(ms_centre_offset(uv)) * 0.70710678;
}

// `ang`: angle around the centre in radians, 0 to the right,
// counter-clockwise, in (-pi, pi].
fn ms_ang(uv: vec2<f32>) -> f32 {
    let d = ms_centre_offset(uv);
    return atan2(d.y, d.x);
}

// `hue_shader`: MilkDrop's slowly drifting corner colours (from its
// `ShowToUser`), interpolated across the image. Corner i (0 top-left,
// 1 top-right, 2 bottom-left, 3 bottom-right) is
//   s = 0.6 + 0.3 * sin(time * 30 * (0.0143, 0.0107, 0.0129)
//                       + (3, 1, 6) + i * (21, 13, 9) + rand_preset.wyz),
//   hue = 0.5 + 0.5 * s / max(s.r, s.g, s.b).
fn ms_hue_corner(i: f32) -> vec3<f32> {
    let phase = u.time * 30.0 * vec3<f32>(0.0143, 0.0107, 0.0129)
        + vec3<f32>(3.0, 1.0, 6.0) + i * vec3<f32>(21.0, 13.0, 9.0) + u.rand_preset.wyz;
    let s = 0.6 + 0.3 * sin(phase);
    return 0.5 + 0.5 * s / max(s.r, max(s.g, s.b));
}

fn ms_hue_shader(uv: vec2<f32>) -> vec3<f32> {
    let t = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let top = mix(ms_hue_corner(0.0), ms_hue_corner(1.0), t.x);
    let bottom = mix(ms_hue_corner(2.0), ms_hue_corner(3.0), t.x);
    return mix(top, bottom, t.y);
}
