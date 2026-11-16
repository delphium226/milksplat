---
name: check-preset
description: Use when a Milksplat preset (.splat) was added or changed and needs its safety, reactivity, brightness and frame-cost numbers, its golden re-blessed, or its gallery image rendered, or when the techno test recording is missing.
---

# Checking a Milksplat preset

Every preset needs the same short set of numbers before it is committed. The user wants testing kept SHORT, so don't run the full suite for a preset change.

## Test audio

The harness reads raw audio: interleaved little-endian f32 stereo at 48 kHz.
- **Recording location:** keep it at `.test-audio/techno.raw` in the repo root. That folder is gitignored, so it survives temp clean-ups. NEVER commit a recording.
- **Generate a stand-in if it's missing:** `python .claude/skills/check-preset/scripts/make_standin_audio.py` writes a synthetic 32 s, 128 bpm techno track there. It has an 11 s quiet breakdown, then the drop.
  - Say in the report that the stand-in was used.
  - Its reactivity scores run a little higher than real music.
- **Other recordings:** `MS_TEST_AUDIO=<path>` overrides the default for the script.

## The checks

`bash .claude/skills/check-preset/scripts/check.sh <preset-stem> [step]` runs from the repo or worktree root. The steps:

| Step | What it runs | Pass |
|---|---|---|
| `safety` | Techno, 1800 ticks at 480x270, at 1x and 5x reactivity | max delta ≤ 0.06 and fewer than 3 flashes/s, at both levels |
| `rank` | `rank_reactivity` at 1x, this preset only | score ≥ ~45 |
| `look` | 900 ticks at 960x540, PNGs at ticks 300, 600 and 900 written to `.test-audio/out/<stem>/` | mean luminance ≥ ~0.10; pick the gallery frame from these |
| `bench` | Frame cost at 1920x1080 | ≤ 2 ms (prefer ≤ 1.5); re-run a noisy number |
| `golden` | Re-blesses the golden on the GPU, then runs its golden test, `bundled_set_is_complete` and `no_late_flashing` | all pass; view the golden PNG |
| `all` (default) | Every step above | |

- **Gallery image:** copy the chosen frame to `docs/images/<stem>.png` (960x540).
- **Kick frame:** also send the chosen frame to the user as `<stem>-a.png`.
- **Before committing code:** run `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` once.
- **WARP:** never use WARP (`MS_FORCE_WARP`) for these numbers.

## When a check fails

- **Max delta or flashes over the limit:** turn whole-screen throbs into local or partial pulses.
- **`no_late_flashing` fails at 64x36:** scale the flare by resolution, so small pictures soften whole-picture kicks. Also look for things that vanish abruptly, such as a reused slot or a particle respawning; fade them out instead.
- **Low score:** use the pulse pattern from the new-presets brief, with bass/mid/treb pulses driving a visible per-hit response. Don't rely on `beat`, and don't double-smooth the bands.
