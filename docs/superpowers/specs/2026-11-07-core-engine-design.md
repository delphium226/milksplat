# Milksplat — Core Engine Design (Sub-project 1)

Date: 2026-11-07
Status: Draft for review

## 1. Purpose and context

Milksplat is a high-performance, audio-reactive music visualizer in the tradition of MilkDrop: a feedback buffer warped by small per-frame and per-vertex programs that react to the music, with waveforms, shapes and shaders drawn on top, and smooth blends between presets.

**Audience:** music listeners. They download it, run it while music plays on their PC, and it looks great with no setup.

**Release:** public, Windows 10/11 first. Code stays portable (no Windows APIs outside clearly isolated modules) so macOS and Linux can follow.

**Licence:** Milksplat's code and bundled presets are dual-licensed MIT OR Apache-2.0.

### 1.1 The v1 release is four sub-projects

| # | Sub-project | Depends on |
|---|---|---|
| 1 | **Core engine** (this spec) | — |
| 2 | Settings overlay (egui) | 1 |
| 3 | In-app preset editor | 1, 2 |
| 4 | `.milk` importer (incl. HLSL → WGSL translation) | 1 |

Each gets its own spec, plan and implementation. This spec covers sub-project 1 only, but its preset model is deliberately MilkDrop-shaped so that sub-project 4 is a translator, not a second engine.

### 1.2 Success criteria

- Sustained 60+ fps at 1080p on a mid-range GPU (GTX 1660 / RX 6600 class) with the default mesh, and at 4K with automatic internal-resolution scaling.
- Preset switches and blends never drop a frame because of shader compilation.
- Visuals line up with audible sound within one frame after A/V offset is set correctly.
- No crash or permanent hang from: invalid preset code or shaders, audio device changes, silence, GPU device loss, minimize, sleep/resume.
- Preset motion is independent of display refresh rate (60 / 120 / 144 Hz look the same).

### 1.3 Out of scope for this sub-project

Settings overlay UI, in-app editor, `.milk` import, HDR output, BPM tracking, macOS/Linux builds, installer and code signing.

## 2. Stack

- Rust (stable), Cargo workspace.
- `wgpu` (DX12 primary, Vulkan fallback) with WGSL shaders, validated by `naga`.
- `cpal` for WASAPI loopback capture; direct `windows` crate calls for device-change notifications (`IMMNotificationClient`), keep-awake (`SetThreadExecutionState`) and Bluetooth endpoint detection.
- `realfft` for the FFT.
- `winit` for windowing and input.
- `rtrb` (or equivalent lock-free SPSC ring) between audio and render threads.
- `serde` + `toml` for presets and config; `notify` for preset hot reload; `tracing` for logging; `image` for texture loading.

## 3. Architecture

### 3.1 Crates

| Crate | Responsibility | Depends on |
|---|---|---|
| `ms-audio` | WASAPI loopback capture, device-change handling, downmix to stereo, timestamped packets into a ring buffer | — |
| `ms-analysis` | Audio history, FFT, waveform, `bass/mid/treb` (+`_att`), beat detection, A/V offset | — |
| `ms-expr` | Expression language: lexer, parser, compiler to bytecode, scalar VM and batched VM | — |
| `ms-preset` | `.splat` file format: parse, validate, defaults; preset library (scan, favourites, bans, shuffle history) | `ms-expr` |
| `ms-render` | wgpu renderer: feedback chain, warp mesh, waves, shapes, borders, motion vectors, blur, composite, text, textures; device-loss recovery | — |
| `ms-engine` | Runs presets: variable environment, per-tick execution order, preset blending, transitions, fixed-tick clock; turns preset state into render commands | `ms-expr`, `ms-preset`, `ms-render`, `ms-analysis` |
| `milksplat` | App binary: window, event loop, render thread, hotkeys, config, logging, crash handling, safe mode | all |

`ms-render` does not know about presets or the expression language; it takes plain data (mesh UVs, draw lists, uniform blocks, compiled pipelines). `ms-engine` is the only crate that knows both sides.

Later sub-projects add `ms-ui` (overlay and editor) and `ms-milk` (importer) without changing the crates above.

### 3.2 Threads

- **Main thread:** winit event loop only. Forwards input, resize, focus, DPI and close events to the render thread over a channel. Never blocks on rendering, so window drags and dialogs don't freeze the visuals.
- **Audio thread:** cpal's capture callback. Downmixes to stereo, stamps the packet with its capture time and pushes into the SPSC ring. Never blocks, never allocates.
- **Render thread:** owns the wgpu device and the engine. Each display frame: drain the audio ring into analysis history, advance the fixed-tick simulation as many ticks as are due, composite, present.
- **Loader thread:** parses and compiles presets, validates WGSL with naga, creates pipelines inside wgpu error scopes, and performs one hidden warm-up draw into an offscreen target before handing the preset to the render thread as "ready".
- **Worker pool** (rayon or scoped threads): batched per-vertex and per-point evaluation, split across cores.

### 3.3 Clock and frame loop

- Simulation runs on a fixed tick, 60 Hz by default (a preset may declare a different `sim_rate`; the clock follows the preset on screen, and during a blend keeps the outgoing preset's rate until the blend completes). Per tick: audio analysis at audio time, preset `frame` code, `vertex` code, warp pass, overlay drawing, blur.
- Composite runs every display frame at display rate, reading the latest feedback buffer. Shader uniforms receive exact (not tick-quantized) `time`.
- If the render thread falls behind by more than 4 ticks, extra ticks are dropped (not run in a burst), and the quality governor (§6.6) is told.
- Present mode: Mailbox if available, else Fifo; wgpu maximum frame latency 1. VRR displays are used as-is.

## 4. Preset format (`.splat`)

One TOML file per preset. Unknown keys produce a warning, not an error. Missing values take defaults.

```toml
version = 1

[meta]
name = "Ink Bloom"
author = "Rhett"
license = "MIT OR Apache-2.0"

[params]
decay = 0.98
zoom = 1.0
warp = 0.5
feedback_format = "rgba8"   # or "fp16"
mesh = [64, 48]             # columns, rows; capped by quality settings
sim_rate = 60
wave_mode = 2
# ... all writable variables from §5.3 may be given starting values here

[code]
init = "..."
frame = "..."
vertex = "..."

[[wave]]          # up to 4
enabled = true
samples = 512
init = "..."
frame = "..."
point = "..."

[[shape]]         # up to 4
enabled = true
sides = 6
instances = 1
init = "..."
frame = "..."

[shader]
warp = """ WGSL """   # optional; built-in default if absent
comp = """ WGSL """   # optional; built-in fixed-function path if absent
```

Textures are referenced by name from shaders. Lookup order: the preset's own folder, then the user `Presets\textures\`, then the bundled `textures\`.

## 5. Expression language (`ms-expr`)

### 5.1 Semantics (EEL2-compatible)

- All values are `f64`. `0` is false; any other value is true.
- A program is a sequence of expressions separated by `;`. The value of a block is its last expression.
- Operators: `+ - * / % ^ (power)`, unary `-` and `!`, comparisons `== != < <= > >=`, logical `&& ||` (short-circuit), bitwise `& |` (on values truncated to integers), assignment `=` and compound `+= -= *= /= %=`.
- Division by zero and modulo by zero yield `0`.
- `==` and `!=` compare with tolerance `1e-5` (absolute).
- Control flow: `if(c, a, b)`, `loop(n, body)`, `while(body)` (repeats while the body's value is true). Every loop is capped: one program execution may perform at most 1,048,576 loop iterations in total, and all of one preset's code in one tick (init, frame, every vertex, every wave frame and point, every shape instance) shares a budget of 1,048,576 iterations; exceeding either stops the loop (once the tick budget is spent, every further loop in that tick exits at once, and the tick still completes) and logs a warning once per preset.
- Built-in functions: `sin cos tan asin acos atan atan2 sqrt pow exp log log10 abs sign min max floor ceil int sqr rand sigmoid band bor bnot equal above below`, plus `megabuf(i)` and `gmegabuf(i)` (each 1,048,576 entries; out-of-range reads return 0 and writes are ignored).
- Variable names are case-insensitive. Unknown names become user variables initialised to 0.
- `rand(n)` uses a per-preset seeded generator so render tests are deterministic.

### 5.2 Variable lifetime and execution order

- **User variables** persist across ticks for the life of the preset instance.
- **Order per tick:**
  1. Read-only inputs are written.
  2. Writable variables are reset to the preset's `[params]` values.
  3. `frame` code runs.
  4. Its `q1`–`q32` are snapshotted.
  5. `vertex` code runs once per mesh point. Each point starts from the post-frame values of the writable variables and the `q` snapshot. User variables written in `vertex` code carry from one point to the next, in row-major order, matching MilkDrop.
  6. Each wave and shape runs its own `frame` code, starting from the `q` snapshot, followed by `point` code for waves.
  7. Shaders receive the `q` snapshot.
- **`init`** runs once when the preset loads, after `[params]` are applied; it may set user variables and `q` values used as the initial snapshot.
- **`reg00`–`reg99`** and `gmegabuf` are global and survive preset changes.

### 5.3 Preset variables

**Read-only:**
- **Time:** `time`, `fps`, `frame`, `progress` (0→1 over the preset's display time).
- **Audio:** `bass`, `mid`, `treb`, `bass_att`, `mid_att`, `treb_att`, `beat`, `beat_strength`.
- **Mesh and aspect:** `meshx`, `meshy`, `aspectx`, `aspecty`, `pixelsx`, `pixelsy`.
- **In `vertex` code only:** `x`, `y`, `rad`, `ang`.
- **In waves:** `sample`, `value1`, `value2`.
- **In shapes:** `instance`.

**Writable (motion):**
- `zoom`, `zoomexp`, `rot`, `warp`
- `cx`, `cy`, `dx`, `dy`, `sx`, `sy`
- `decay`

**Writable (main wave):**
- `wave_mode`, `wave_r`, `wave_g`, `wave_b`, `wave_a`
- `wave_x`, `wave_y`, `wave_mystery`
- `wave_usedots`, `wave_thick`, `wave_additive`, `wave_brighten`

**Writable (borders):** `ob_size`, `ob_r`, `ob_g`, `ob_b`, `ob_a`, `ib_size`, `ib_r`, `ib_g`, `ib_b`, `ib_a`.

**Writable (motion vectors):** `mv_x`, `mv_y`, `mv_dx`, `mv_dy`, `mv_l`, `mv_r`, `mv_g`, `mv_b`, `mv_a`.

**Writable (composite effects):**
- `echo_zoom`, `echo_alpha`, `echo_orient`
- `gamma`, `invert`, `solarize`, `brighten`, `darken`, `darken_center`

**Writable (sampling and blur):**
- `texture_wrap`
- `b1n`, `b2n`, `b3n`, `b1x`, `b2x`, `b3x`, `b1ed`

**Writable (wave-local):** `x`, `y`, `r`, `g`, `b`, `a`.

**Writable (shape-local):**
- `x`, `y`, `rad`, `ang`, `sides`
- `r`, `g`, `b`, `a`, `r2`, `g2`, `b2`, `a2`
- `border_r`, `border_g`, `border_b`, `border_a`
- `additive`, `thickoutline`, `textured`, `tex_zoom`, `tex_ang`

**Shared:** `q1`–`q32` (frame → vertex, waves, shapes, shaders); `t1`–`t8` (per wave or shape, `init` → `frame` → `point`); `reg00`–`reg99` (global).

### 5.4 Compiler and VMs

- Source → tokens → AST → constant-folded AST → register-based bytecode.
- **Scalar VM:** executes per-frame, per-wave/shape-frame and `init` code.
- **Batched VM:** executes `vertex` and wave `point` code over structure-of-arrays lanes. Each opcode runs across a block of 256 points before the next opcode. Mesh rows are partitioned across worker threads.
  - Because user variables carry from point to point (§5.2), the compiler checks whether a `vertex` program reads a user variable before writing it in the same execution. If so, the program runs on the scalar VM in row-major order; otherwise it is batched.
  - The batched VM must produce bit-identical results to the scalar VM for any program it accepts.
- Errors carry line and column: parse errors, unknown functions, wrong argument counts. A preset with a code error loads with that block disabled and the error logged; it does not fail to load.

## 6. Rendering (`ms-render`, driven by `ms-engine`)

### 6.1 Resources

- **Feedback buffers:** a ping-pong pair at internal resolution.
  - Format per preset: `rgba8unorm` (default, and the format imported presets use) or `rgba16float`.
  - In fp16 mode the warp pass clamps output to [0, 64] and replaces NaN/Inf with 0.
- **Blur chain:** three levels at successively half resolution, two-pass separable Gaussian. Only the levels a preset's shaders reference are computed.
- **Noise textures:** generated at startup from a fixed seed.
  - 2D: `noise_lq`, `noise_mq`, `noise_hq`.
  - 3D: `noisevol_lq`, `noisevol_hq`.
- **User textures:** PNG/JPG, using MilkDrop's sampler-name prefixes for wrap/clamp and linear/point filtering (`fw_`, `fc_`, `pw_`, `pc_`). Names `rand00`–`rand15` pick a random file from `textures\random\`.

### 6.2 Per-tick order

1. Engine runs code (§5.2), producing mesh UVs and per-point colour/alpha.
2. **Warp pass:** draw the mesh sampling the previous feedback buffer into the next, using the preset `warp` shader or the built-in warp with `decay`.
3. **Overlay drawing into the feedback buffer, in this order:**
   1. motion vectors
   2. custom shapes
   3. custom waves
   4. main waveform
   5. outer border, then inner border
4. **Blur:** the required blur levels.

### 6.3 Per-display-frame composite

Composite the latest feedback buffer to the swapchain, using the preset `comp` shader or the built-in fixed-function path:
- video echo
- gamma
- brighten, darken and solarize
- invert
- darken-center

Then draw on-screen text.

### 6.4 Waves and shapes

- **Main waveform:** MilkDrop's 8 modes (0–7), with dots, thick lines, additive blending and brighten.
- **Custom waves:** up to 512 points each, drawn as lines or dots, optionally additive, coloured per point.
- **Shapes:**
  - 3–100 sides, centre-to-edge colour gradient and optional border;
  - optionally textured from the feedback buffer using `tex_zoom` and `tex_ang`;
  - up to 1024 instances, each running the shape's `frame` code with its own `instance` value.

### 6.5 Shader interface

- **Uniform block:** one fixed std140 block, identical for warp and composite shaders:
  - time, fps, frame, progress;
  - band levels with `_att` and beat;
  - `q1`–`q32` packed as 8 × vec4;
  - `rand_frame` (vec4, new each tick) and `rand_preset` (vec4, fixed per preset);
  - `roam_cos` and `roam_sin` (vec4 each);
  - `texsize` (xy = size, zw = 1/size), `aspect`;
  - blur min/max ranges.
- **Bindings:**
  - `sampler_main`;
  - `sampler_blur1`–`3`;
  - the noise textures;
  - up to 16 user textures, bound by name in a fixed slot order recorded when the preset is compiled.
- **Shader prelude:** user shader source is a function body wrapped by a prelude that declares all of the above. This is the layout the `.milk` importer will target.

### 6.6 Quality governor

- Tracks a 2-second moving average of GPU frame time.
- Steps internal resolution scale down (to 50% minimum) when over 90% of the frame budget, and back up when under 60%, with hysteresis of 3 seconds per step.
- Mesh size is capped by the quality setting (max 192×144).
- Requests the high-performance GPU adapter on systems with more than one.

### 6.7 Robustness

- An uncaptured-error handler logs instead of panicking.
- All pipeline creation is wrapped in error scopes. On failure, the preset's shader is replaced by the built-in default, the error is logged, and the preset still runs.
- **Device lost:** recreate adapter, device and every GPU resource. The current preset is reloaded from its cached compiled form.
  - If the loader was preparing a preset when the device was lost (its shader compile or warm-up draw), that preset is blamed; otherwise, if the loss happened within 2 s of a preset becoming active, that preset is. The blamed preset (never the built-in default) is added to a blocklist in the config and skipped from then on; its load is not retried.
- **Surface lost or outdated:** reconfigure the surface.
- **Zero-size surface** (minimized): skip configure and rendering; the simulation pauses.

## 7. Transitions and preset selection (`ms-engine`)

### 7.1 Blending

- During a blend, both presets run their code each tick.
- **Mesh:** the two presets' warp-mesh UVs and colours are mixed per vertex. Each vertex's weight comes from a randomly chosen blend pattern:
  - uniform
  - radial in or out
  - directional sweep
  - plasma noise
- **Drawing:**
  - Both presets' warp shaders draw into the one shared feedback chain, weighted by the per-vertex blend.
  - Both presets' waves and shapes are drawn with alpha scaled by the global blend progress.
  - Both composite shaders draw, weighted, into the swapchain.
- **Duration:** default 2.7 s, configurable.

### 7.2 Hard cuts

- Optional; on by default.
- A beat with `beat_strength` above a configurable threshold, at least 10 s after the previous change, cuts instantly to the next preset.

### 7.3 Selection

- **Auto-advance:** default 60 s ± 25% random. `L` locks the current preset.
- **Shuffle:** a history of the last 50 presets. `←` steps back through history. Presets shown in the last 30% of the library are not repeated.
- **Favourites:** shown twice as often.
- **Banned presets:** never shown.
- **Loading:** the next preset is loaded and warmed up on the loader thread before its turn. A manually chosen preset that isn't ready keeps the current preset on screen until it is.

## 8. Audio (`ms-audio`, `ms-analysis`)

### 8.1 Capture

- WASAPI loopback, shared mode, on the default render endpoint or the device named in config.
- Each channel layout is downmixed to stereo:
  - front left/right are kept;
  - centre and LFE are added to both channels at −3 dB;
  - surround channels are added to their side at −3 dB.
- The stream restarts on default-device change, device removal, or stream error, retrying after 0.5, 1, 2 and 5 s, then every 5 s.
- No resampling; analysis works in Hz using the device's sample rate.

### 8.2 Silence and missing audio

- **Silence:** if no packet arrives for more than 50 ms, the analysis history is filled with zeros up to the current time.
- **Hint:** after 10 s of continuous silence, a short on-screen hint appears ("No audio detected — is something playing?"). It disappears when audio returns.

### 8.3 Analysis

Runs once per simulation tick.

- **History:** at least 1 s of stereo, filled by draining the whole ring each frame.
- **Analysis window:** ends at `now − av_offset`.
- **Waveform:** 576 samples per channel for wave code.
- **Spectrum:**
  - FFT of 2048 samples with a Hann window, mono mix;
  - 512 bins of magnitude, log-scaled for display;
  - exposed to waveform modes and as a 512×1 texture `sampler_spectrum`.
- **Bands:**
  - bass 20–250 Hz, mid 250–4000 Hz, treb 4000–16000 Hz;
  - each band's energy is divided by its own long-term average (exponential, ~5 s time constant, floored to avoid division by near-zero), so ≈1.0 means "normal for this track";
  - the instant values are lightly smoothed (≈ 50 ms);
  - the `_att` values are smoothed more (≈ 400 ms);
  - constants are tuned to match MilkDrop's response on reference tracks.
- **Beat detection:**
  - spectral flux of the bass band against an adaptive threshold (mean plus k × standard deviation over the last 1 s), with a 150 ms refractory period;
  - `beat` is 1 on the tick of an onset and 0 otherwise;
  - `beat_strength` is the flux-to-threshold ratio, decaying over 200 ms.

### 8.4 A/V offset

- Set per device in config, and adjusted in 10 ms steps with `[` and `]`.
- Defaults: 0 ms for wired outputs, 200 ms for endpoints Windows reports as Bluetooth.

## 9. App shell (`milksplat`)

### 9.1 Windowing

- **Fullscreen:** borderless, toggled with `F`, `Alt+Enter` or double-click.
- **Monitor:** the fullscreen monitor is chosen by config, and `M` cycles through monitors.
- **Window state:** size, position and the last monitor are saved on exit. Per-monitor DPI is respected.
- **Keep awake:** the display is kept awake while fullscreen with non-silent audio.

### 9.2 Hotkeys

| Key | Action |
|---|---|
| `→` / `←` | Next / previous preset |
| `R` | Random preset |
| `L` | Lock / unlock |
| `*` / `Del` | Favourite / ban the current preset |
| `Space` | Hard cut to next |
| `[` / `]` | A/V offset −/+ 10 ms |
| `Q` | Cycle quality |
| `N` | Show preset name |
| `` ` `` | fps / debug readout |
| `H` | Help |
| `F` | Fullscreen |
| `M` | Next monitor |
| `Esc` | Leave fullscreen, or quit when windowed |

### 9.3 Files

- **Config:** `%APPDATA%\Milksplat\config.toml`, created with defaults on first run.
  - Invalid or unreadable values are logged and replaced by defaults; the app always starts.
  - Contains: audio device, A/V offsets, timings, quality, favourites, bans, blocklist, window state.
- **Presets:**
  - bundled presets in `presets\` next to the exe;
  - user presets in `%USERPROFILE%\Documents\Milksplat\Presets\`;
  - both scanned recursively;
  - hot reload: a modified `.splat` file is reloaded within 0.5 s, and if it's the current preset it is swapped in without a blend.
- **Logs:** `%LOCALAPPDATA%\Milksplat\logs\`, rotated daily, 7 files kept.

### 9.4 Crash handling and safe mode

- **Panic hook:** writes a crash report next to the logs (last 200 log lines, GPU adapter and driver, the current preset path, the app version) and shows a native message box.
- **Crash detection:**
  - A `running` marker records the current preset.
  - If the marker exists at startup, the last run crashed.
  - That preset is then skipped and a notice shown.
- **`--safe` flag:** starts with the built-in default preset, no user shaders and the default audio device.

## 10. Testing

- **`ms-expr`:**
  - Unit tests for every operator, function and EEL quirk: divide-by-zero, fuzzy equality, short-circuit, loop cap, megabuf bounds, case-insensitivity.
  - Property tests: random programs give identical results on the scalar and batched VMs.
  - The vertex dependency check is tested with programs that carry values from point to point.
- **`ms-analysis`:**
  - synthetic signals: a sine at 60/1000/8000 Hz lands in the correct band;
  - a click track at 120 BPM yields beats within ±1 tick;
  - silence decays without NaN;
  - sample rates 44.1/48/96 kHz give consistent band values.
- **`ms-audio`:** downmix matrix tests for 2.0, 5.1 and 7.1; gap-filling on missing packets.
- **`ms-preset`:** round-trip parse/serialise; defaults; unknown-key warnings; invalid files report errors with locations.
- **`ms-render` / `ms-engine`** golden-image tests:
  - headless wgpu renders each bundled preset for 120 ticks with fixed recorded audio and fixed seeds;
  - results are compared with reference PNGs using a per-pixel tolerance;
  - they run on WARP when no GPU is present;
  - blending, device-loss recovery (simulated by recreating the device mid-run) and shader-fallback each have a dedicated test.
- **Benchmarks** (criterion):
  - VM throughput, scalar and batched;
  - a full tick at default and maximum mesh;
  - a regression of more than 10% fails review.
- **Manual release checklist:**
  - unplug or switch the default device;
  - Bluetooth headphones;
  - sleep/resume;
  - minimize/restore;
  - multi-monitor with mixed DPI and refresh rate;
  - a 144 Hz display;
  - exclusive-mode app;
  - 4K at full quality on target hardware.

## 11. Release packaging

- One `milksplat.exe`:
  - built `--release` with LTO;
  - static CRT;
  - built-in shaders, fonts and the default preset embedded.
- Portable zip containing the exe, `presets\` with ~10 original native presets (also used as render-test fixtures), `textures\`, `LICENSE-MIT`, `LICENSE-APACHE` and a short README.
- Installer and code signing are deferred; the README explains the SmartScreen prompt.

## 12. Risks

- **Batched VM correctness:** mitigated by scalar/batched property tests and the automatic scalar fallback.
- **MilkDrop look fidelity:** band scaling, blend patterns and the 8-bit feedback behaviour need tuning against reference recordings. Golden images lock in the result once tuned.
- **Shader compile hitches on some drivers:** mitigated by loader-thread compile and warm-up draws. Measured with the frame-time readout during testing.
- **HLSL translation for sub-project 4:** naga has no HLSL front end, so a custom translator will be needed. This doesn't affect this sub-project, but the shader prelude (§6.5) is the contract it must target.
