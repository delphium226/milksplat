#!/bin/bash
# Short checks for one Milksplat preset. Run from the repo (or worktree) root.
# usage: bash .claude/skills/check-preset/scripts/check.sh <stem> [safety|rank|look|bench|golden|all]
set -u
stem="${1:?usage: check.sh <preset-stem> [step]}"
step="${2:-all}"
export PATH="$HOME/.cargo/bin:$PATH"
[ -f "presets/$stem.splat" ] || { echo "no presets/$stem.splat here ($(pwd))"; exit 1; }

# The recording lives in the MAIN checkout's .test-audio (worktrees share it).
main_root="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
audio="${MS_TEST_AUDIO:-$main_root/.test-audio/techno.raw}"
if [ ! -f "$audio" ]; then
    echo "no recording at $audio; generating the synthetic stand-in"
    python "$main_root/.claude/skills/check-preset/scripts/make_standin_audio.py" "$audio" || exit 1
fi
echo "audio: $audio"
out="$main_root/.test-audio/out/$stem"
mkdir -p "$out"

tune() { # <reactivity> <ticks> <size> <shots> [bench]
    MS_TUNE_ONLY="$stem" MS_TUNE_REACTIVITY="$1" MS_TUNE_TICKS="$2" MS_TUNE_SIZE="$3" \
    MS_TUNE_SHOTS="$4" MS_TUNE_AUDIO="$audio" MS_TUNE_OUT="$out" MS_TUNE_BENCH="${5:+1}" \
        cargo test -q --release -p ms-engine --test presets_e2e tune_presets -- --ignored --nocapture 2>&1 |
        grep -E "luminance|delta|flash|ms/tick|error|panicked|-->" | head -20
}

run() {
    case "$1" in
    safety)
        echo "== safety 1x"; tune 1 1800 480x270 1800
        echo "== safety 5x"; tune 5 1800 480x270 1800 ;;
    rank)
        echo "== rank_reactivity 1x"
        MS_TUNE_ONLY="$stem" MS_TUNE_AUDIO="$audio" MS_RANK_LEVELS=1 \
            cargo test -q --release -p ms-engine --test presets_e2e rank_reactivity -- --ignored --nocapture 2>&1 |
            grep -E "^$stem |score|panicked|error" | head -5 ;;
    look)
        echo "== look 960x540 (PNGs in $out)"; tune 1 900 960x540 300,600,900 ;;
    bench)
        echo "== bench 1920x1080"; tune 1 60 480x270 60 bench ;;
    golden)
        test_name="preset_${stem//-/_}"
        echo "== bless $test_name"
        MS_BLESS=1 cargo test -q --release -p ms-engine --test presets_e2e "$test_name" 2>&1 | grep -E "test result|panicked"
        echo "== tests"
        cargo test -q --release -p ms-engine --test presets_e2e -- "$test_name" bundled_set_is_complete no_late_flashing 2>&1 |
            grep -E "^test |test result|panicked"
        echo "golden: crates/ms-engine/tests/golden/$test_name.png" ;;
    all) for s in safety rank look bench golden; do run "$s"; done ;;
    *) echo "unknown step $1"; exit 1 ;;
    esac
}
run "$step"
