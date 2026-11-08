// Composite stage wrapper: one full-screen triangle that calls `user_comp`
// (the preset's composite body, or the built-in one) per fragment.
//
// The result is sanitised (NaN/Inf to 0) and clamped to [0, 1]; alpha is 1.
// Layers are mixed with the blend constant (see composite.rs).

struct CompVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle covering the target: (-1,-1), (3,-1), (-1,3).
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> CompVsOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var out: CompVsOut;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    // v = 0 at the top (texture space).
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: CompVsOut) -> @location(0) vec4<f32> {
    let c = user_comp(in.uv, ms_rad(in.uv), ms_ang(in.uv), ms_hue_shader(in.uv));
    return vec4<f32>(min(ms_sanitise(c), vec3<f32>(1.0)), 1.0);
}
