//! The hotkey table (spec §9.2): keys to [`Action`]s.

use winit::keyboard::{Key, ModifiersState, NamedKey};

/// What a hotkey asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Next,
    Prev,
    Random,
    ToggleLock,
    Favourite,
    Ban,
    HardCut,
    /// Adjust the A/V offset by this many milliseconds.
    AvOffset(i32),
    CycleQuality,
    ShowName,
    ToggleFps,
    ToggleHelp,
    ToggleFullscreen,
    NextMonitor,
    /// Leave fullscreen, or quit when windowed.
    EscapeKey,
}

impl Action {
    /// Whether holding the key down should repeat the action. Toggles and
    /// one-shot actions ignore key repeats.
    pub fn repeats(self) -> bool {
        matches!(self, Action::Next | Action::Prev | Action::AvOffset(_))
    }
}

/// The §9.2 table, for the help panel: (keys, what they do).
pub const HELP: &[(&str, &str)] = &[
    ("Right / Left", "next / previous preset"),
    ("R", "random preset"),
    ("L", "lock / unlock"),
    ("* / Del", "favourite / ban the current preset"),
    ("Space", "hard cut to next"),
    ("[ / ]", "A/V offset -/+ 10 ms"),
    ("Q", "cycle quality"),
    ("N", "show preset name"),
    ("`", "fps / debug readout"),
    ("H", "help"),
    ("F, Alt+Enter, double-click", "fullscreen"),
    ("M", "next monitor"),
    ("Esc", "leave fullscreen, or quit"),
];

/// Maps a logical key (with the current modifiers) to its action. Letters
/// ignore case and Shift (Shift is also how `*` is typed); any chord with
/// Ctrl, Alt or the Windows key is not a hotkey, except `Alt+Enter`, and
/// except a character typed with Ctrl+Alt: Windows reports AltGr as
/// Ctrl+Alt, and AltGr is how `[`, `]` and others are typed on many
/// layouts.
pub fn map_key(key: &Key, mods: ModifiersState) -> Option<Action> {
    let altgr = mods.control_key() && mods.alt_key() && !mods.super_key();
    let chord = mods.control_key() || mods.alt_key() || mods.super_key();
    if chord && !(altgr && matches!(key, Key::Character(_))) {
        let alt_only = mods.alt_key() && !mods.control_key() && !mods.super_key();
        return (alt_only && *key == Key::Named(NamedKey::Enter))
            .then_some(Action::ToggleFullscreen);
    }
    match key {
        Key::Named(named) => match named {
            NamedKey::ArrowRight => Some(Action::Next),
            NamedKey::ArrowLeft => Some(Action::Prev),
            NamedKey::Delete => Some(Action::Ban),
            NamedKey::Space => Some(Action::HardCut),
            NamedKey::Escape => Some(Action::EscapeKey),
            _ => None,
        },
        Key::Character(s) => {
            let mut chars = s.chars();
            let (Some(c), None) = (chars.next(), chars.next()) else {
                return None;
            };
            match c.to_ascii_lowercase() {
                'r' => Some(Action::Random),
                'l' => Some(Action::ToggleLock),
                '*' => Some(Action::Favourite),
                '[' => Some(Action::AvOffset(-10)),
                ']' => Some(Action::AvOffset(10)),
                'q' => Some(Action::CycleQuality),
                'n' => Some(Action::ShowName),
                '`' => Some(Action::ToggleFps),
                'h' => Some(Action::ToggleHelp),
                'f' => Some(Action::ToggleFullscreen),
                'm' => Some(Action::NextMonitor),
                _ => None,
            }
        }
        _ => None,
    }
}
