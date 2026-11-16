---
name: new-presets
description: Use when the user asks for more Milksplat presets ("add another 10 presets", "go crazy"), or asks to rework, restyle or brighten existing bundled presets.
---

# Adding or reworking Milksplat presets in parallel lanes

Presets are built by background agents working in parallel, about 5 presets per agent, each in its own git worktree. You plan the batch, brief the lanes, then merge, check and ship.

## 1. Plan the batch
- **Pick themes:** choose names unmistakably different from everything in `presets/` (list it).
  - Use the user's style: dark backgrounds, bright saturated graphics, showpieces.
  - Original ideas only; no recreations of existing games or shows.
- **Split into lanes:** about 5 presets per lane.
  - Give each preset a name, a stem and a 2–3 line visual brief, including what kicks do.
  - Name any existing preset it must not resemble.
- **Test recording:** make sure `.test-audio/techno.raw` exists in the main checkout. If it doesn't, run `python .claude/skills/check-preset/scripts/make_standin_audio.py`, and tell the user the numbers come from the synthetic stand-in.

## 2. Launch the lanes
Use one `Agent` call per lane, all in ONE message, with:
- `isolation: "worktree"`, `run_in_background: true`, `model: "opus"`, `subagent_type: "general-purpose"`.
- A prompt with these parts, in this order:
  1. Tell it to read the brief first and follow it exactly: `.claude/skills/new-presets/preset-brief.md`. For reworks, add `rework-brief.md` after it.
  2. Its lane id (`lane-<batch><letter>`) for scratch files under `.test-audio/`.
  3. Its numbered presets.
  4. The names the other lanes are doing, so it avoids overlap.

Tell the user, in a few lines, which presets each lane is building.

## 3. As each lane reports
- **Kick frames:** send that lane's `<stem>-a.png` files with `SendUserFile` straight away, captioned with the names.
- **Merge:** the first lane fast-forwards (`git merge --ff-only <branch>`). Later lanes conflict in the test lists and the README gallery, so:
  ```bash
  git merge --no-edit <branch>
  python .claude/skills/new-presets/scripts/merge_lanes.py
  ```
  The script resolves both files, re-pairs the gallery, fixes the array lengths and updates the README's preset count. Then commit the merge.
- **Clean up:** once a lane is merged, run `git worktree remove <path>` and `git branch -d <branch>`.
- **Reports:** don't believe a report's numbers blindly. If it used the stand-in audio, say so.

## 4. Ship (after the last lane)
1. Check the README intro's preset count. The merge script sets it; after a single-lane batch, run the script once anyway.
2. Run the new presets' golden tests and `bundled_set_is_complete`. Keep it short:
   `cargo test -p ms-engine --test presets_e2e -- bundled_set_is_complete <stem_with_underscores>...`
3. Copy the README and each new `.splat` to `dist/milksplat-<version>-win64/` (the presets go in its `presets/` folder; skip `default.splat`).
4. Commit the merge, then fast-forward `main` to `core-engine` and push both: `git checkout main && git merge --ff-only core-engine && git push origin main core-engine && git checkout core-engine`.
5. Confirm with `git fetch && git rev-parse origin/main HEAD`.

## Common mistakes
- **Reserved names:** using `b1x`–`b3x`, `b1n`–`b3n` or `dx` as preset variables. They are built-ins and get overwritten every frame.
- **Gallery merges:** merging half-empty last rows by hand. Use the script, which re-pairs the whole table.
- **Recording in git:** committing `.test-audio/` or any `.raw` file.
- **Long-running checks:** running the full test suite or WARP for a preset batch.
