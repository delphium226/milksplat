// Warp stage wrapper: draws the warp mesh and calls `user_warp` (the preset's
// warp body, or the built-in one) per fragment.
//
// Fragment alpha is the layer weight; the pipeline blends colour as
// src * srcAlpha + dst and leaves the target's alpha (cleared to 1) alone.
// The result is multiplied by the per-vertex colour and sanitised (NaN/Inf
// to 0, clamped to [0, 64]).

struct WarpVsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    // rgb = per-vertex colour, a = layer weight.
    @location(2) color: vec4<f32>,
};

struct WarpVsOut {
    @builtin(position) clip: vec4<f32>,
    // Where this point samples the previous frame (the warped UV).
    @location(0) uv: vec2<f32>,
    // The point's own position in texture space (the unwarped UV).
    @location(1) uv_orig: vec2<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: WarpVsIn) -> WarpVsOut {
    var out: WarpVsOut;
    out.clip = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.uv_orig = vec2<f32>(in.pos.x * 0.5 + 0.5, 0.5 - in.pos.y * 0.5);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: WarpVsOut) -> @location(0) vec4<f32> {
    let c = user_warp(
        in.uv,
        in.uv_orig,
        ms_rad(in.uv_orig),
        ms_ang(in.uv_orig),
        ms_hue_shader(in.uv_orig),
    );
    return vec4<f32>(ms_sanitise(c * in.color.rgb), in.color.a);
}
