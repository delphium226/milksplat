// Full-target bilinear copy, used to keep the feedback image across a resize
// or a format change. Values are sanitised like the warp output; alpha is 1.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

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

fn sanitise(v: vec3<f32>) -> vec3<f32> {
    let exponent = bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u);
    let bad = exponent == vec3<u32>(0x7f800000u);
    return clamp(select(v, vec3<f32>(0.0), bad), vec3<f32>(0.0), vec3<f32>(64.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(sanitise(textureSample(src, src_sampler, in.uv).rgb), 1.0);
}
