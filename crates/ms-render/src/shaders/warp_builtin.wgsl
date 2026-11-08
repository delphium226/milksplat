// Built-in warp body (the inside of `user_warp`, see prelude.wgsl and
// warp_stage.wgsl): the previous frame at the warped UV, linear filtering and
// wrap addressing, times decay. The stage wrapper multiplies by the vertex
// colour and sanitises.
return GetMain(uv) * u.decay;
