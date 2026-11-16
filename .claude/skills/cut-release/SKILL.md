---
name: cut-release
description: Use when the user asks to release, publish or tag a new Milksplat version (e.g. "create the v0.2.0 release"), or to rebuild the dist zip, exe or screensaver (.scr) for download.
---

# Cutting a Milksplat release

Releases go to GitHub `delphium226/milksplat` as a tag `vX.Y.Z`, with one asset, `milksplat-X.Y.Z-win64.zip`. The zip holds, at its top level: `milksplat.exe`, `milksplat.scr` (the same exe renamed), `presets/` (no `default.splat`), `textures/`, both licences and the README.

**Publishing is outward-facing.** Before tagging, confirm the version number with the user, unless they named it.

## Steps
1. **Start clean:** be on `core-engine`, with everything committed, and `main` equal to `core-engine`.
2. **Bump the version** wherever it appears:
   - the workspace `version` in the root `Cargo.toml`;
   - the six `ms-*` path dependencies below it;
   - `README.md` Running step 1 (`milksplat-X.Y.Z-win64.zip`).

   Then `cargo build -p milksplat` to refresh `Cargo.lock`. Find stragglers with `grep -rn "<old version>" Cargo.toml crates/*/Cargo.toml README.md`.
3. **Commit:** `chore: release X.Y.Z`. Fast-forward `main` and push both branches (`git push origin main core-engine`).
4. **Package:** `powershell -NoProfile -File scripts/package.ps1`. It runs the workspace tests once, builds the release exe, stages `dist/milksplat-X.Y.Z-win64/`, zips it and checks for the static CRT. If the tests have just passed on this commit, add `-SkipTests`.
   - Check the presets count: in `dist/milksplat-X.Y.Z-win64/presets` it must equal `presets/*.splat` minus `default`.
5. **Notes:** write the release notes to a temp file. Use the previous release's notes as the template: `gh release view <prev tag> --repo delphium226/milksplat --json body --jq .body`. Use the same sections:
   - Download, with the unzip-and-run line and the SmartScreen "More info → Run anyway" note;
   - What's in it, with the preset count first;
   - Requirements.

   Lead with what changed since the previous release (`git log <prev tag>..HEAD --oneline`), grouped as new presets, reworked presets and app changes.
6. **Publish:**
   ```bash
   gh release create vX.Y.Z dist/milksplat-X.Y.Z-win64.zip --repo delphium226/milksplat \
     --target main --title "Milksplat X.Y.Z" --notes-file "$TEMP/notes.md"
   ```
7. **Verify:**
   - Download the asset into an empty temp folder (`gh release download vX.Y.Z --repo delphium226/milksplat --dir <dir>`) and `cmp` it with the local zip.
   - `git fetch --tags && git rev-parse vX.Y.Z^{commit}` must equal `origin/main`.
   - `https://github.com/delphium226/milksplat/releases/latest` must redirect to the new tag.
8. **Report:** give the user the release URL, the zip size and the preset count. Say that the screensaver's exit-on-input can only be checked by hand.

## Common mistakes
- **Zip layout:** zipping the staging folder itself, which nests everything one level down, instead of its contents. `package.ps1` already zips the contents.
- **README filename:** forgetting the README's versioned zip filename. The link itself uses `/releases/latest` and needs no change.
- **Other repos:** pushing anywhere other than `delphium226/milksplat`, or adding test recordings (`.test-audio/`, `*.raw`) to a commit.
- **gh notes:** using `--generate-notes`. The user's release notes are hand-written prose.
