# Milksplat

Milksplat is an original music visualizer for Windows: GPU feedback visuals
that react to whatever your PC is playing. It is inspired by the classic
MilkDrop-style visualizers. It captures the audio from your default output
device, so just start your music and run it.

## Gallery

| | |
|---|---|
| ![Infinite Coil](docs/images/infinite-coil.png) | ![Turing Reef](docs/images/turing-reef.png) |
| **Infinite Coil**: an endless spiral zoom | **Turing Reef**: coral that grows and melts with the music |
| ![Lumen Cathedral](docs/images/lumen-cathedral.png) | ![Ebru Tide](docs/images/ebru-tide.png) |
| **Lumen Cathedral**: a rose window pouring out light | **Ebru Tide**: live paper marbling |
| ![Tesseract Drift](docs/images/tesseract-drift.png) | |
| **Tesseract Drift**: a 4D hypercube turning inside out | |

These are single frames; every preset moves and changes with the music.

## Requirements

- Windows 10 or 11
- A GPU with DirectX 12 or Vulkan support

## Running

1. Unzip the release anywhere.
2. Run `milksplat.exe`.
3. Play some music.

Windows may warn that the app is unrecognised; click More info → Run anyway.
(The release is not code-signed yet.)

Command line options: `--presets <dir>` uses another preset folder,
`--safe` starts in safe mode, `--exit-after <seconds>` quits after a delay and
`--log-fps` logs the frame rate.

## Hotkeys

| Key | Action |
|---|---|
| Right / Left | next / previous preset |
| R | random preset |
| L | lock / unlock the current preset |
| * / Del | favourite / ban the current preset |
| Space | hard cut to the next preset |
| [ / ] | A/V offset -/+ 10 ms |
| Q | cycle quality |
| N | show the preset name |
| ` | fps / debug readout |
| H | help |
| F, Alt+Enter, double-click | fullscreen |
| M | next monitor |
| Esc | leave fullscreen, or quit |

Letters ignore case. Other Ctrl, Alt or Windows-key chords are not hotkeys,
except that AltGr characters (typed as Ctrl+Alt) work, for keyboard layouts
where `[` and `]` need AltGr.

## Where things live

| What | Where |
|---|---|
| Bundled presets and textures | `presets\` and `textures\` next to `milksplat.exe` |
| Your presets | `%USERPROFILE%\Documents\Milksplat\Presets\` (created on first run) |
| Config | `%APPDATA%\Milksplat\config.toml` |
| Logs and crash reports | `%LOCALAPPDATA%\Milksplat\logs\` |

Drop `.splat` files into your presets folder; edits to preset files reload
live.

## Safe mode

If Milksplat crashed during the last run, it skips the preset that was
playing and shows a notice. `--safe` starts with only the built-in default
preset, no user shaders and the default audio device. Use it if a preset or
setting stops the app from starting.

## Writing presets

A preset is a TOML `.splat` file: parameters, plus code blocks that run every
frame (and per point for waves). For example, a wave that traces the
waveform:

```toml
version = 1

[meta]
name = "Scope"

[params]
decay = 0.9

[code]
frame = '''
ob_a = 0.1 + 0.3 * min(bass, 2) / 2;
'''

[[wave]]
enabled = true
samples = 256
additive = 1
point = '''
x = sample;
y = 0.5 + 0.3 * value1;
'''
```

Notes:

- Code blocks use an EEL2-style expression language; audio arrives as
  variables such as `bass`, `mid`, `treb` and `time`.
- Shaders are WGSL function bodies, not whole shaders: Milksplat supplies the
  surrounding prelude.
- Inside per-pixel branches, sample textures with `textureSampleLevel`.
- Saved changes hot reload; errors fall back to the previous good version and
  are logged.

The full format and language are specified in sections 4 and 5 of
[the design spec](docs/superpowers/specs/2026-11-07-core-engine-design.md).
`presets\` has working examples.

## Licence

MIT OR Apache-2.0, at your option. See `LICENSE-MIT` and `LICENSE-APACHE`.
