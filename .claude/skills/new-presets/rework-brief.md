# Rework brief (read preset-brief.md first, then this)

This time you are REWORKING existing bundled presets (your prompt names them and says what is wrong with each), not adding new ones. Everything in `preset-brief.md` still applies (hard rules, photosensitivity, frame cost, reactivity pulse pattern, testing kept SHORT, commit attribution, no push/merge/app launch, kick frames `<stem>-a.png`, dist copy), with these differences:

- **Keep each preset's identity:** same file, same id, same `name`, same core idea (what the user would recognise it by). Rewrite as much of the code as you need to.
- **Fix the look:** typical problems are lit, washed-out backgrounds (slate blue, purple, green, orange haze) and near-static motion. Make each a showpiece in the current style: a DARK background (near black, deep colour at most) with bright, saturated foreground graphics, richer detail and depth. Mean luminance still >= ~0.10 at 960x540.
- **Make it strongly reactive:** `rank_reactivity` score >= ~45 at 1x, with visible per-hit response (the user complained about static presets).
- **Test lists:** the presets are already in `BUNDLED`, the golden macro and `LONG_RUN`; do not add them again. If a preset's `LONG_RUN` tick count is 1200, raise it to 3600 like the newer presets.
- **Golden:** re-bless it (`MS_BLESS=1`) on the GPU and view it.
- **Gallery:** overwrite `docs/images/<name>.png` (960x540) with a new representative frame, and update that preset's caption cell in the README Gallery table to describe the new look: `**Name**: one line`. Edit only the caption text in place; don't move rows.
- **Commit:** one per preset, message like `feat(presets): rework <Name>, <short description>`.
- **Report:** per preset, before → after in one line, what reacts, max delta 1x/5x, score, mean luminance, ms/frame; concerns; the 5 `-a.png` paths.
