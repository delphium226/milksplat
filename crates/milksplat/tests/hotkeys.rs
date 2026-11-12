//! The §9.2 hotkey table (plus `Alt+Enter`) maps to the right `Action`, and
//! anything else maps to nothing.

use milksplat::config::Config;
use milksplat::hotkeys::{map_key, reactivity_notice, Action, HELP};
use winit::keyboard::{Key, ModifiersState, NamedKey, SmolStr};

fn ch(s: &str) -> Key {
    Key::Character(SmolStr::new(s))
}

fn none() -> ModifiersState {
    ModifiersState::empty()
}

#[test]
fn every_table_row_maps() {
    let rows: Vec<(Key, ModifiersState, Action)> = vec![
        (Key::Named(NamedKey::ArrowRight), none(), Action::Next),
        (Key::Named(NamedKey::ArrowLeft), none(), Action::Prev),
        (ch("r"), none(), Action::Random),
        (ch("l"), none(), Action::ToggleLock),
        (ch("*"), ModifiersState::SHIFT, Action::Favourite),
        (ch("*"), none(), Action::Favourite), // numpad `*`
        (Key::Named(NamedKey::Delete), none(), Action::Ban),
        (Key::Named(NamedKey::Space), none(), Action::HardCut),
        (ch("["), none(), Action::AvOffset(-10)),
        (ch("]"), none(), Action::AvOffset(10)),
        (ch("q"), none(), Action::CycleQuality),
        (ch("n"), none(), Action::ShowName),
        (ch("`"), none(), Action::ToggleFps),
        (ch("h"), none(), Action::ToggleHelp),
        (ch("f"), none(), Action::ToggleFullscreen),
        (ch("m"), none(), Action::NextMonitor),
        (Key::Named(NamedKey::Escape), none(), Action::EscapeKey),
        (Key::Named(NamedKey::ArrowUp), none(), Action::Reactivity(1)),
        (
            Key::Named(NamedKey::ArrowDown),
            none(),
            Action::Reactivity(-1),
        ),
        (
            Key::Named(NamedKey::Backspace),
            none(),
            Action::ResetReactivity,
        ),
    ];
    for (key, mods, want) in rows {
        assert_eq!(map_key(&key, mods), Some(want), "{key:?} with {mods:?}");
    }
}

#[test]
fn letters_ignore_case_and_shift() {
    assert_eq!(
        map_key(&ch("R"), ModifiersState::SHIFT),
        Some(Action::Random)
    );
    assert_eq!(map_key(&ch("F"), none()), Some(Action::ToggleFullscreen));
    assert_eq!(
        map_key(&ch("H"), ModifiersState::SHIFT),
        Some(Action::ToggleHelp)
    );
}

#[test]
fn alt_enter_toggles_fullscreen() {
    assert_eq!(
        map_key(&Key::Named(NamedKey::Enter), ModifiersState::ALT),
        Some(Action::ToggleFullscreen)
    );
}

#[test]
fn altgr_characters_map() {
    // Windows reports AltGr as Ctrl+Alt; `[` and `]` need it on many layouts.
    let altgr = ModifiersState::CONTROL | ModifiersState::ALT;
    assert_eq!(map_key(&ch("["), altgr), Some(Action::AvOffset(-10)));
    assert_eq!(map_key(&ch("]"), altgr), Some(Action::AvOffset(10)));
    assert_eq!(
        map_key(&ch("]"), altgr | ModifiersState::SHIFT),
        Some(Action::AvOffset(10))
    );
    // Not with the Windows key, and not for named keys.
    assert_eq!(map_key(&ch("["), altgr | ModifiersState::SUPER), None);
    assert_eq!(map_key(&Key::Named(NamedKey::ArrowRight), altgr), None);
}

#[test]
fn unmapped_keys_give_none() {
    let unmapped: Vec<(Key, ModifiersState)> = vec![
        (Key::Named(NamedKey::Enter), none()),
        (Key::Named(NamedKey::Tab), none()),
        (Key::Named(NamedKey::F1), none()),
        (ch("a"), none()),
        (ch("z"), none()),
        (ch("1"), none()),
        (ch("ff"), none()),
        (ch(""), none()),
        // Ctrl / Alt / Super chords are not hotkeys (except Alt+Enter).
        (ch("r"), ModifiersState::CONTROL),
        (ch("f"), ModifiersState::ALT),
        (Key::Named(NamedKey::ArrowRight), ModifiersState::CONTROL),
        (Key::Named(NamedKey::ArrowUp), ModifiersState::CONTROL),
        (Key::Named(NamedKey::Backspace), ModifiersState::ALT),
        (Key::Named(NamedKey::Escape), ModifiersState::SUPER),
        (
            Key::Named(NamedKey::Enter),
            ModifiersState::CONTROL | ModifiersState::ALT,
        ),
        (Key::Dead(None), none()),
    ];
    for (key, mods) in unmapped {
        assert_eq!(map_key(&key, mods), None, "{key:?} with {mods:?}");
    }
}

/// `Ctrl+Shift+F12` crashes the render thread on purpose, in debug builds
/// only; nothing else maps to it.
#[test]
fn debug_crash_chord() {
    let f12 = Key::Named(NamedKey::F12);
    let want = cfg!(debug_assertions).then_some(Action::DebugCrash);
    assert_eq!(
        map_key(&f12, ModifiersState::CONTROL | ModifiersState::SHIFT),
        want
    );
    for mods in [
        none(),
        ModifiersState::CONTROL,
        ModifiersState::SHIFT,
        ModifiersState::CONTROL | ModifiersState::SHIFT | ModifiersState::ALT,
    ] {
        assert_eq!(map_key(&f12, mods), None, "F12 with {mods:?}");
    }
}

#[test]
fn reactivity_steps_repeat_and_reset_does_not() {
    assert!(Action::Reactivity(1).repeats());
    assert!(Action::Reactivity(-1).repeats());
    assert!(!Action::ResetReactivity.repeats());
}

#[test]
fn reactivity_keys_are_in_the_help() {
    assert!(HELP.iter().any(|(k, _)| k.contains("Up / Down")));
    assert!(HELP.iter().any(|(k, _)| k.contains("Backspace")));
}

/// Up/Down move the shown preset's level in quarter steps within
/// 0.25..=5; Backspace (and stepping back to 1x) removes its entry.
#[test]
fn reactivity_hotkeys_change_the_stored_level() {
    let mut c = Config::default();
    assert_eq!(c.step_preset_reactivity("a.splat", 1), 1.25);
    assert_eq!(c.step_preset_reactivity("a.splat", 1), 1.5);
    assert_eq!(c.preset_reactivity("a.splat"), 1.5);
    for _ in 0..20 {
        c.step_preset_reactivity("a.splat", 1);
    }
    assert_eq!(c.preset_reactivity("a.splat"), 5.0);
    for _ in 0..20 {
        c.step_preset_reactivity("b.splat", -1);
    }
    assert_eq!(c.preset_reactivity("b.splat"), 0.25);
    // Back to 1x: no entry.
    for _ in 0..3 {
        c.step_preset_reactivity("b.splat", 1);
    }
    assert_eq!(c.preset_reactivity("b.splat"), 1.0);
    assert!(!c.preset_reactivity.contains_key("b.splat"));
    // Backspace.
    c.set_preset_reactivity("a.splat", 1.0);
    assert!(c.preset_reactivity.is_empty());
}

#[test]
fn reactivity_notice_text() {
    assert_eq!(reactivity_notice(1.5, 1.0), "Reactivity 1.5x");
    assert_eq!(reactivity_notice(1.0, 1.0), "Reactivity 1x");
    assert_eq!(reactivity_notice(0.25, 1.0), "Reactivity 0.25x");
    // A global multiplier other than 1 shows the effective level too.
    assert_eq!(
        reactivity_notice(1.5, 0.5),
        "Reactivity 1.5x (effective 0.75x)"
    );
    assert_eq!(
        reactivity_notice(3.0, 2.0),
        "Reactivity 3x (effective 5x)"
    );
}
