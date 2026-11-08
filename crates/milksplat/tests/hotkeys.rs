//! The §9.2 hotkey table (plus `Alt+Enter`) maps to the right `Action`, and
//! anything else maps to nothing.

use milksplat::hotkeys::{map_key, Action};
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
        (Key::Named(NamedKey::ArrowUp), none()),
        (Key::Named(NamedKey::ArrowDown), none()),
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
