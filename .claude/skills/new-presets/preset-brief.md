# Milksplat preset lane brief

You are adding about 5 new bundled presets to Milksplat, a Rust/wgpu audio-reactive music visualizer. Your prompt names the presets and your lane id.

You work in a git worktree of the repo, on your own branch. Stay in the worktree and commit there. If `core-engine` is ahead of your HEAD and you have no commits yet, run `git merge --ff-only core-engine`. The first cargo build in a fresh worktree is slow; that is normal.

**Paths:**
- `<main>` means the main checkout: `git rev-parse --git-common-dir`, then its parent.
- Keep your scratch files in `<main>/.test-audio/<lane-id>/`. That folder is gitignored.

## Learning the format
- **Examples:** read 2–3 recent presets to learn the format and the current reactivity idioms: `presets/starfield-warp.splat`, `presets/pulse-grid.splat`, `presets/saturns-rings.splat` and `presets/voronoi-crystals.splat`.
- **Spec:** the format is in sections 4 and 5 of `docs/superpowers/specs/2026-11-07-core-engine-design.md`.
- **Reactivity rulings:** R76–R79 in `docs/superpowers/core-engine-decisions.md`.

## The look
- **Showpieces:** be bold, spectacular and inventive.
- **Colour:** dark backgrounds (near black, deep colour at most) with bright, saturated foreground graphics. This is the user's standing preference.
- **Distinctness:** each preset must look unmistakably different from every existing preset (list `presets/`).
- **Originality:** don't recreate an existing game, film or show.

## Reactivity
Every preset must be strongly music-reactive at the default 1x, because the user dislikes static presets.
- **Pulse pattern:** build pulses from the scaled `bass`, `mid` and `treb`.
  - Threshold 1 (silence gives 0), soft limit `x/(1+x/2)`, attack about 0.6, decay about 0.1.
  - Don't rely on `beat`, and don't double-smooth or hard-`min` the bands.
- **Visible per-hit response:** drive size, camera push, flares, bursts, waves and colour.
- **Score target:** about 45 or more from `rank_reactivity` at 1x.

## Hard rules
- **Shader bodies:** user shaders are function bodies, `user_warp(uv, uv_orig, rad, ang, hue_shader) -> vec3f` and `user_comp(uv, rad, ang, hue_shader) -> vec3f`.
- **Branches:** inside per-pixel branches, use `textureSampleLevel`.
- **q variables:** q1–q32 are `u.q[..]`.
- **Reserved names:** `b1x`, `b2x`, `b3x`, `b1n`, `b2n`, `b3n` and `dx` are built-ins, overwritten every frame. Never use them as your own variables.
- **Phases:** NEVER multiply absolute `time` by an audio-driven speed. Accumulate the phase in frame code.
- **Photosensitivity:**
  - Max per-tick full-frame luminance delta ≤ 0.06, and no full-frame flashing at 3 Hz or more.
  - Check it at 1x AND 5x.
  - Prefer local or partial-frame pulses to whole-screen throbs.
  - `no_late_flashing` runs at 64x36, so scale whole-picture flares down on small pictures. Fade things out rather than letting them vanish, for example when a slot is reused.
- **Frame cost:** ≲ 2 ms at 1080p on an RTX 3090; prefer ≤ 1.5 ms.
  - Shape instances cost about 0.75 µs each on the CPU.
  - Wave point code runs on the slow scalar VM.
  - Big unrolled shader loops spill registers.
  - Another lane may be benchmarking at the same time, so re-run a noisy number.
- **Brightness:** mean luminance ≥ about 0.10 at 960x540, and don't start from an empty screen.
- **naga:** it rejects a constant `let v = vec2f(...)` used through swizzles; use scalars instead.
- **Ghost copies:** if the built-in warp with low decay leaves ghost copies, a one-line custom warp fixes it.
- **Feedback buffers:** simulations in the feedback buffer must survive the buffer size changing.
- **Engine code:** don't change it. Edit only presets, the test lists, goldens, `docs/images` and the README.

## Testing (keep it SHORT; the user insists)
Use the check-preset script from your worktree root: `bash .claude/skills/check-preset/scripts/check.sh <stem> [step]`. Read `.claude/skills/check-preset/SKILL.md` for the steps and pass limits.
- **Audio:** the script uses the shared recording in `<main>/.test-audio/techno.raw`, generating the synthetic stand-in if it is missing. Say in your report which one you used. NEVER commit a recording.
- **Test lists:** add each preset to `BUNDLED`, the `preset_tests!` macro and `LONG_RUN` (3600 ticks) in `crates/ms-engine/tests/presets_e2e.rs`, and bump the array lengths.
- **What not to run:** don't run the full suite and don't use WARP.
- **Lints:** at the very end, run `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` once.

## Deliverables per preset
- **Gallery image:** `docs/images/<stem>.png` at 960x540, a good representative frame.
- **README rows:** append the image row and then the caption row at the end of the `## Gallery` 2-column table, just before "These are single frames; …".
  - Captions are `**Name**: one line`.
  - Keep the file's line endings.
  - Another lane is appending in parallel; the merge script fixes the pairing later.
- **Dist copy:** copy the .splat to `<main>/dist/milksplat-<version>-win64/presets/`, if that folder exists.
- **Commit:** one commit per preset, like `feat(presets): <Name>, <short description>`, ending with the Co-Authored-By line from your instructions.
- **Kick frame:** save one at a kick to `<main>/.test-audio/<stem>-a.png`.
- **Off limits:** do NOT push, merge into other branches or launch the app.

## Report back (concise)
- Branch, worktree path and commit hashes.
- Which audio you used.
- Per preset:
  - a one-line description and what reacts;
  - max delta at 1x / 5x and flashes per second;
  - the reactivity score, mean luminance and ms per frame.
- Concerns.
- The `-a.png` paths.
