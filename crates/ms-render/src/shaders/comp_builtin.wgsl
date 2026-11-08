// Built-in composite, following MilkDrop's fixed-function (shader-less)
// composite. Inserted after the prelude; the built-in comp body is
// `return ms_comp_builtin(uv);`.
//
// Steps, in this order (c is the colour, all flags are blended linearly by
// their value so that 0 = off, 1 = on and preset transitions stay smooth):
//
// 1. Video echo: a second copy of the image, zoomed about the centre by
//    echo_zoom and flipped by echo_orient (0 none, 1 left-right, 2 up-down,
//    3 both; rounded and clamped to 0..3), mixed in by echo_alpha:
//      echo_uv = (uv - 0.5) * flip / max(echo_zoom, 0.001) + 0.5
//      c = mix(GetMain(uv), GetMain(echo_uv), echo_alpha)
// 2. Gamma: c = pow(max(c, 0), 1 / max(gamma, 0.001)), then clamped to
//    [0, 1]. (MilkDrop brightens by re-adding the image `gamma` times, which
//    is roughly c * gamma; the pow form is this project's ruling.)
// 3. brighten: c = mix(c, sqrt(c), brighten)          (MilkDrop: square-root
//                                                      filter)
//    darken:   c = mix(c, c * c, darken)              (squaring filter)
//    solarize: c = mix(c, 4 * c * (1 - c), solarize)  (peaks at mid-grey)
// 4. invert:   c = mix(c, 1 - c, invert)
// 5. darken_center: MilkDrop draws a black disc at the centre with alpha 3/32
//    at the middle, fading linearly to 0 at radius 0.05 (clip units,
//    aspect-corrected); here
//      c = c * (1 - darken_center * 3/32 * max(0, 1 - r / 0.05)),
//    r = length(ms_centre_offset(uv)).
//
// Every step keeps c in [0, 1] for [0, 1] inputs; with default uniforms
// (gamma 1, echo_alpha 0, all flags 0) the result is the main image.

fn ms_comp_builtin(uv: vec2<f32>) -> vec3<f32> {
    // 1. Video echo.
    let orient = u32(clamp(round(u.echo_orient), 0.0, 3.0));
    let flip = vec2<f32>(
        select(1.0, -1.0, (orient & 1u) != 0u),
        select(1.0, -1.0, (orient & 2u) != 0u),
    );
    let echo_uv = (uv - 0.5) * flip / max(u.echo_zoom, 0.001) + 0.5;
    var c = mix(GetMain(uv), GetMain(echo_uv), clamp(u.echo_alpha, 0.0, 1.0));

    // 2. Gamma.
    c = clamp(pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / max(u.gamma, 0.001))), vec3<f32>(0.0), vec3<f32>(1.0));

    // 3. Brighten, darken, solarize.
    c = mix(c, sqrt(c), clamp(u.brighten, 0.0, 1.0));
    c = mix(c, c * c, clamp(u.darken, 0.0, 1.0));
    c = mix(c, 4.0 * c * (1.0 - c), clamp(u.solarize, 0.0, 1.0));

    // 4. Invert.
    c = mix(c, 1.0 - c, clamp(u.invert, 0.0, 1.0));

    // 5. Darken centre.
    let r = length(ms_centre_offset(uv));
    c = c * (1.0 - clamp(u.darken_center, 0.0, 1.0) * (3.0 / 32.0) * max(0.0, 1.0 - r / 0.05));
    return c;
}
